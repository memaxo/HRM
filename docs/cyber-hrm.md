Below is a **concrete, end‑to‑end implementation plan** that stays as close as possible to the current HRM codebase while extending it for automated exploit discovery.  File / class names that already exist in the repository are written in **bold**; new ones are *italicised*.

---

## 1  Environment‑Integration Architecture

| Layer                                            | Responsibility                                                        | Key APIs / Pseudo‑code | Notes |
| ------------------------------------------------ | --------------------------------------------------------------------- | ---------------------- | ----- |
| **gym‑compatible wrapper**<br>*cybersec\_env.py* | Presents a single, unified RL interface to HRM (`reset()`, `step(a)`) | \`\`\`python           |       |
| class CyberSecEnv(gym.Env):                      |                                                                       |                        |       |

````
def reset(self, bin_path):
    self.proc = TargetSandbox(bin_path)
    self.sym = AngrHelper(bin_path)
    return self._encode_state()

def step(self, action_tokens: np.ndarray):
    low_lvl_cmd = self._decode_action(action_tokens)
    result = self.proc.execute(low_lvl_cmd)
    self.sym.update(result)
    reward = self._compute_reward(result)
    obs = self._encode_state()
    done = self._is_terminal(result)
    return obs, reward, done, {}``` | • One running binary per HRM **batch element**.<br>• `TargetSandbox` selects AFL++, libFuzzer, or raw QEMU instrumentation depending on the action. |
````

\| *TargetSandbox* | Executes the SUT under QEMU‑user or Docker and returns traces | `run(input) → {crash, coverage_bitmap, stdout}` |
\| *AngrHelper* | Maintains symbolic state, path constraints, taint information | `update(trace)` returns incremental coverage, SAT/UNSAT of new constraints, etc. |
\| **HRM <-> Env bridge** | Instantiates `CyberSecEnv` for every training worker. Inside **train\_batch()** replace the current `(inputs, labels)` loop by RL roll‑outs when `dataset_mode="rl"` | Uses the same “carry” mechanism: the env state tensor drives `current_data["inputs"]`; the model’s chosen actions are written into `current_data["labels"]` (see §3) |

### 1.1  State Representation  → `_input_embeddings()`

| Security object         | Pre‑tensor serialisation                         | Token shape fed to HRM                             |
| ----------------------- | ------------------------------------------------ | -------------------------------------------------- |
| Coverage bitmap (64 kB) | Bit‑packed → uint8 vector length = 64 k          | `[B, 64 k]`                                        |
| CFG (≤1 024 nodes)      | Adjacency matrix (bool) + 1‑hot basic‑block type | Flatten to `[1024 × (1 + feat_dim)]` then zero‑pad |
| Register / memory slice | byte sequence → hex‑byte tokens (0‑255)          | variable‑length, capped (see §3)                   |
| Path constraints        | **angr** SMT clauses → hashed to 128‑bit id      | 128‑bit split into 16 bytes                        |

These vectors are concatenated and **right‑padded to a fixed `seq_len`** (e.g. `131 072` tokens) so *puzzle\_dataset.py* stays unchanged.  The YAML in **cfg\_pretrain.yaml** simply sets:

```yaml
seq_len: 131072      # max observation length
vocab_size: 258      # 0=PAD, 1=EOS, 2‑257 = bytes
```

(258 leaves room for action tokens—see below).

### 1.2  Action Space

| Primitive          | Token payload                   | Example                    |
| ------------------ | ------------------------------- | -------------------------- |
| `MUTATE_BYTE`      | offset (4 B) + new\_value (1 B) | flip offset 1234 to `0x41` |
| `CROSSOVER_INPUTS` | id\_A (2 B), id\_B (2 B)        | AFL‑style splice           |
| `SOLVE_CONSTRAINT` | constraint\_hash (16 B)         | ask Z3 to satisfy path     |
| `FOLLOW_BRANCH`    | block\_id (2 B)                 | depth‑first exploration    |
| `ALLOC_HEAP`       | size (4 B)                      | trigger heap feng‑shui     |
| `SEND_PACKET`      | len (2 B) + bytes               | network targets            |

Every primitive is mapped to **one base opcode (1 byte)**; variable payload bytes follow.  The **lm\_head replacement (§3)** emits a *variable‑length* action sequence that is parsed until EOS.

---

## 2  Dataset‑Construction Pipeline

### 2.1  Synthetic Generation (`dataset/build_cyber_dataset.py`)

1. **Seed program synthesis**

   * Use [csmith](https://github.com/csmith-project/csmith) or [llvm‑morpheus](https://github.com/GaloisInc/morpheus) to generate random C programs.
   * Instrument with sanitisers to auto‑label the vulnerability type.
2. **Vulnerability injection**

   * Template‑based patches for classic bugs (`strcpy`, double‑free, off‑by‑one, …).
3. **Ground‑truth exploit**

   * Deterministic exploit script generated via `pwntools` + `angr` (`angr.exploration_techniques.Oppologist`).

Result is saved exactly like the puzzle datasets:

* **inputs.npy** – bytecode of the *vulnerable binary* (or LLVM IR) padded to `seq_len`.
* **labels.npy** – *reference exploit action sequence*, tokenised exactly as §1.2.
* **puzzle\_identifiers.npy** – program ID (enable augmentation).
* **group\_indices.npy** – keep one binary per group so HRM can **loop internally** until it halts.

Aim for **≥1 000 binaries per vulnerability class**.
Augmentation parallels ARC:

* Byte‑level mutations (`dd if=/dev/urandom`) ↔ colour permutations.
* ASLR re‑randomisations ↔ translational shifts.
* Dihedral transform analogue: reorder functions / basic blocks with `--function-sections` + `ld --randomize-sections`.

### 2.2  Curriculum

1. **Phase 0 – Unit tests**: single‑basic‑block stack overflows (input length ≤ 32).
2. **Phase 1 – Simple BOF**: linear CFG, no heap, no ASLR.
3. **Phase 2 – Heap feng‑shui**: `malloc/free` patterns, small‑bin attack.
4. **Phase 3 – CTF‑level**: PIE, NX, partial RELRO, network I/O.
5. **Phase 4 – “Wild”**: real OSS‑Fuzz corpora, Chrome bugs etc.

Encode the phase as an extra token prepended to the observation; *puzzle\_dataset* already supports identifiers.

---

## 3  Model‑Side Changes

### 3.1  Embedding Layer

```python
# models/hrm/security_embeddings.py
class ByteEmbed(nn.Module):
    def __init__(self, vocab_size=258, hidden=512):
        self.table = CastedEmbedding(vocab_size, hidden, init_std=1/hidden**0.5, cast_to=torch.bfloat16)
```

Replace **embed\_tokens** in **HierarchicalReasoningModel\_ACTV1\_Inner** with `ByteEmbed`; keep `RoPE`.

### 3.2  Latent State (`z_H`, `z_L`)

*Current shape*: `[B, seq_len, hidden]`
*New*: keep shape, *but reserve channel slots*:

| Channel index range | Meaning                                   |
| ------------------- | ----------------------------------------- |
|  0‑63               | coverage bitmap (after linear projection) |
| 64‑127              | CFG summary                               |
| 128‑…               | raw byte context                          |

A single `nn.Linear(hidden, hidden, bias=False)` “projection mixer” can be inserted at the start of **\_input\_embeddings()** to interleave these features.

### 3.3  Output Heads

```python
class SecurityHeads(nn.Module):
    def __init__(self, hidden, vocab_size, num_actions):
        self.action_head  = CastedLinear(hidden, vocab_size, bias=False)   # generates opcodes+bytes
        self.conf_head    = CastedLinear(hidden, 1, bias=True)            # exploit success prob
        self.sat_head     = CastedLinear(hidden, 1, bias=True)            # constraint satisfaction
```

Replace every occurrence of `lm_head` by `action_head`.  The two scalar heads are averaged into the existing **ACTLossHead** via new keys `"conf_loss", "sat_loss"`.

### 3.4  Q‑Learning Modifications

*Q‑halt* now predicts **“terminate exploit chain”** rather than “stop reasoning”.
Reward shaping proposal:

| Event                           | Reward         | Feeding into        |
| ------------------------------- | -------------- | ------------------- |
| New coverage bitmap bit         | +0.1           | `target_q_continue` |
| Unique crash                    | +5             | `q_continue_logits` |
| Exploit works (EIP/RIP control) | +20; terminate | `q_halt_logits`     |

*Implementation*
`target_q_continue = reward + γ · max(Q(s', a'))`, already predicted in **HierarchicalReasoningModel\_ACTV1.forward()** – only change the reward tensor.

---

## 4  Training Infrastructure

| Component      | Specification                                                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Isolation      | **Docker‑in‑Docker** with `--cap-drop ALL`, `seccomp`, no network, read‑only image. For kernel‑level fuzzing use **QEMU‑kvm** inside the container.                             |
| Episode length | 8 – 32 HRM steps (`halt_max_steps`) × 2 (L\_cycles) × 2 (H\_cycles)  → ≤ 128 tool invocations.                                                                                  |
| Hardware       | RTX 4090 × 4 GPUs, 128 GB RAM, 24‑core CPU. One env worker ↔ one CPU core.                                                                                                      |
| Throughput     | ≈ 5 k environment steps / s across workers (\~16 × slower than Sudoku). Expect **\~150 GPU‑hours** for 100 k steps (comparable to ARC numbers).                                 |
| Metrics        | `train/uniq_crashes`, `train/coverage`, `train/exploit_rate`, `train/avg_steps`, `train/loss`, `train/q_halt_accuracy`.  Log via W\&B exactly where **pretrain.py** logs today. |

---

## 5  Safety & Evaluation

### 5.1  Containment

1. **Hypervisor boundary**: run each worker VM on a **nested‑virt** host; no shared mounts.
2. **Outbound filter**: `iptables -P OUTPUT DROP` inside guest, allow only instrumentation sockets.
3. **Time‑bomb**: KVM VM auto‑shutoff after N seconds or high CPU to avoid infinite loops.

### 5.2  Benchmarks

| Dataset                                        | Why                                                  |
| ---------------------------------------------- | ---------------------------------------------------- |
| **DARPA Cyber Grand Challenge (CGC) binaries** | Fixed architecture, labelled crashes, ⇒ reproducible |
| **LAVA‑M + LAVA‑X**                            | Synthetic but realistic stacks/heap bugs             |
| **pwn.college “sprout” tier challenges**       | Public, small, varied                                |
| **OSS‑Fuzz regression corpora**                | Real‑world, with ASAN/UBSAN signal                   |

### 5.3  Dual‑use Controls

* “Weights under NDA” licence; require affiliation.
* `--safe-inference` flag masks `SEND_PACKET`, `ALLOC_HEAP` opcodes while still permitting passive analysis.
* Audit log: every generated action chain + SHA‑256 of target binary stored, rotated daily.

---

## 6  Next Steps Checklist

1. **Implement *cybersec\_env.py*** and confirm a single HRM forward pass drives AFL++ for a toy binary.
2. **Extend `dataset/` pipeline** for 1 000 simple BOF samples; verify `pretrain.py` runs 1 epoch without NaNs.
3. **Swap in `SecurityHeads` and remove `lm_head`**; watch `action_head` logits for entropy.
4. **Incrementally raise `halt_max_steps`** and enable real reward; track `uniq_crashes` trending upward.
5. **Add CGC evaluation script** mirroring `evaluate.py`.

With these changes you can **reuse > 90 % of the existing HRM training stack** and start producing research‑grade exploit agents in a matter of weeks rather than months.

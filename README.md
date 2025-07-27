# ATHENA (Automated Tactical Hacking and Exploit Neural Architecture)

## Hierarchical Reasoning Model for Cybersecurity

![](./assets/hrm.png)

ATHENA applies the Hierarchical Reasoning Model (HRM) to autonomous exploit generation and cybersecurity research. Building on HRM's hierarchical and multi-timescale processing capabilities, ATHENA learns to discover and exploit vulnerabilities through strategic reasoning and precise primitive execution.

The system operates through two interdependent modules: a high-level strategy planner that reasons about exploitation approaches (information leakage, memory corruption, control flow hijacking), and a low-level primitive executor that handles detailed operations (gadget selection, payload construction, memory manipulation). This hierarchical approach enables ATHENA to learn complex exploitation techniques from minimal training data, without requiring extensive pre-training or chain-of-thought supervision.

ATHENA demonstrates the potential for AI systems to advance cybersecurity research through automated vulnerability discovery and exploitation, while maintaining strict safety controls and ethical boundaries.

## Quick Start Guide 🚀

### Running on Apple Silicon (M-series) 🍏

```bash
# Install latest PyTorch with Metal Performance Shaders support
pip install torch>=2.3.0 torchvision torchaudio

# Install cybersecurity dependencies
pip install -r requirements-cyber.txt

# Setup ATHENA environment
python scripts/setup_cyber_mvp.py

# Start training on synthetic exploit data
python pretrain.py --config-name=cyber/base_cyber device=mps
```

### Running on CUDA 🐍

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements-cyber.txt

# Setup environment
python scripts/setup_cyber_mvp.py

# Train with CUDA acceleration
python pretrain.py --config-name=cyber/base_cyber device=cuda
```

## Core Dependencies 🔧

```bash
# Core ML and cybersecurity tools
pip install -r requirements-cyber.txt

# Key dependencies include:
# - torch>=2.3.0 (Metal/CUDA support)
# - pwntools (exploit development)
# - capstone (disassembly)
# - gymnasium (RL environments)
# - lief (binary analysis)
```

## W&B Integration 📈

ATHENA uses [Weights & Biases](https://wandb.ai/) for experiment tracking and cybersecurity metrics:

```bash
wandb login
```

## Training Experiments

### Phase 0: Infrastructure Demo 💻

Verify all components work together on synthetic data:

```bash
# Generate synthetic exploit training data
python scripts/setup_cyber_mvp.py

# Quick training run (100 steps)
python pretrain.py --config-name=cyber/base_cyber epochs=1 eval_interval=50
```

*Runtime:* ~5 minutes on M1 Mac

### Phase 1: Stack Overflow Exploitation 🎯

Train ATHENA to perform basic stack buffer overflows:

```bash
# Generate stack overflow training scenarios
python dataset/generators/stack_exploits.py --output-dir data/stack_exploits --num-samples 1000

# Train stack exploitation model
python pretrain.py \
    --config-name=cyber/stack_exploit \
    data_path=data/stack_exploits \
    epochs=5000 \
    eval_interval=500 \
    global_batch_size=64
```

*Runtime:* ~2 hours on RTX 4070, ~4 hours on M1 Max

**Success Metrics:**
- ≥1 EIP overwrite in 50 episodes
- Training loss convergence
- Memory usage < 16GB

### Phase 2: ROP Chain Construction 🔗

Advanced exploitation with Return-Oriented Programming:

```bash
# Generate ROP chain training data
python dataset/generators/rop_chains.py --output-dir data/rop_chains --num-binaries 100

# Train ROP chain generation
python pretrain.py \
    --config-name=cyber/rop_chains \
    data_path=data/rop_chains \
    epochs=10000 \
    eval_interval=1000 \
    global_batch_size=32
```

*Runtime:* ~8 hours on 8x RTX 4090, ~16 hours on M1 Ultra

**Success Metrics:**
- 40% success rate on test binaries
- Average chain length < 20 gadgets
- DEP bypass capability

## Trained Checkpoints 🏆

- [Stack Overflow Exploitation](https://huggingface.co/athena-ai/stack-overflow-v1) - Basic buffer overflow techniques
- [ROP Chain Generation](https://huggingface.co/athena-ai/rop-chains-v1) - Return-oriented programming
- [Multi-Stage Exploits](https://huggingface.co/athena-ai/multi-stage-v1) - Complex exploitation chains

## Evaluation & Testing

### Security Testing 🛡️

ATHENA includes comprehensive security testing:

```bash
# Run security audit
python scripts/security_audit.py

# Test exploit environment isolation
pytest tests/test_sandbox_security.py -v

# Memory budget verification
pytest tests/perf/test_memory_budget.py -v
```

### Exploit Success Evaluation

```bash
# Evaluate on test binaries
python evaluate.py \
    checkpoint=checkpoints/athena-stack-v1.pt \
    test_set=data/test_binaries \
    max_episodes=50

# Generate detailed reports
jupyter notebook eval/exploit_analysis.ipynb
```

## Dataset Preparation

### Synthetic Data Generation

```bash
# Stack overflow scenarios
python dataset/build_stack_dataset.py --difficulty easy --num-samples 1000

# ROP gadget extraction
python dataset/build_rop_dataset.py --binary-dir /usr/bin --max-binaries 100

# Multi-vulnerability scenarios  
python dataset/build_multi_vuln_dataset.py --complexity medium
```

### Real-World Binary Analysis

```bash
# Extract features from real binaries (ethical use only)
python tools/binary_analyzer.py --input-dir ethical_samples/ --output-dir data/real_binaries/

# Note: Only use binaries you own or have explicit permission to analyze
```

## Architecture Components

ATHENA extends HRM with cybersecurity-specific modules:

- **Strategy Planner (H-Level)**: High-level exploitation strategies
- **Primitive Executor (L-Level)**: Low-level exploit primitives  
- **State Encoder**: Program state representation
- **Action Decoder**: Primitive-to-payload conversion
- **Exploit Environment**: Sandboxed binary execution
- **Tool Adapters**: Integration with security tools

## Safety & Ethics 🔒

ATHENA is designed for defensive cybersecurity research:

- **Sandboxed Execution**: All exploits run in isolated containers
- **No Network Access**: Exploit environments are network-isolated
- **Audit Logging**: All actions are logged for security review
- **Ethical Guidelines**: Strict usage policies for responsible research

## Contributing

ATHENA development follows responsible disclosure practices:

1. Security vulnerabilities → private disclosure
2. Feature requests → public GitHub issues
3. Research contributions → peer review process

## Citation 📜

```bibtex
@misc{athena2024,
    title={ATHENA: Automated Tactical Hacking and Exploit Neural Architecture}, 
    author={[Jack Mazac]},
    year={2025},
    note={Built on Hierarchical Reasoning Model foundation},
    url={https://github.com/your-org/athena}
}

@misc{wang2025hierarchicalreasoningmodel,
    title={Hierarchical Reasoning Model}, 
    author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
    year={2025},
    eprint={2506.21734},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2506.21734}, 
}
```

## License & Legal Notice ⚖️

ATHENA is released for educational and defensive cybersecurity research only. Users are responsible for ensuring compliance with all applicable laws and regulations. Unauthorized use for malicious purposes is strictly prohibited.

---

*ATHENA: Advancing cybersecurity through hierarchical AI reasoning*
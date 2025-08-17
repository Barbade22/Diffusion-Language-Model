
# 🚀 Diffusion Language Base Model (From Scratch)

A complete implementation of diffusion-based language modeling with real-time interactive visualization, built from scratch with novel architectural innovations.

## 🎯 Overview
This repository contains a research prototype implementation of a Diffusion Language Model (DiffusionLM) that addresses fundamental limitations of autoregressive language models through parallel iterative text generation.

### The Problem with Traditional Language Models
- **Sequential Bottleneck**: Each token waits for previous ones - no parallelization
- **Error Propagation**: One wrong prediction derails the entire sequence
- **Limited Controllability**: Hard to refine or correct text mid-generation

### Our Diffusion Solution
Instead of generating tokens sequentially, our model refines all tokens simultaneously across multiple denoising steps:

```
Traditional (Sequential):     "The" → "cat" → "sat" → "on" → "mat"
Diffusion (Parallel):         [MASK][MASK][MASK][MASK][MASK] 
                          →   [The][MASK][sat][MASK][mat]
                          →   [The][cat][sat][on][mat]
```

## 🏗️ Architecture & Innovations

### Key Technical Contributions Beyond [LLDM Paper](https://arxiv.org/abs/2502.09992):

1. **Time Embeddings**: Custom sinusoidal time embedding modules for timestep conditioning
2. **Cosine Beta Scheduling**: Smooth noise scheduling for better denoising trajectories  
3. **Memory-Optimized Training**: Gradient checkpointing and efficient data loading
4. **Custom Tokenization**: BPE tokenizer optimized for Marathi language
5. **Real-time Visualization**: Interactive diffusion process demonstration

### Model Specifications:
- **Parameters**: 22M (research prototype)
- **Architecture**: 12-layer Transformer with diffusion conditioning
- **Vocab Size**: 12,000 tokens (optimized for efficiency)
- **Sequence Length**: 256 tokens
- **Diffusion Steps**: 50 (optimized for real-time inference)
  Note: Architecture can be modified using CONFIG inside train.py

## 📁 Repository Structure

```
# Main training script 
├── Pre-Training/
│   ├── base_model/                # Pre-trained model components
│   ├── optimal_checkpoint_epoch_135.pt # Weights cane download from my HF repo 
│   ├── tokenizer_optimal.pkl
│   ├── ui.py                     # Interactive web interface
│   ├── data-Sample.txt           # Training data (Marathi text) example
│   └── train.py                  # Pre-training Script


My check point wait
## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/diffusion-language-model.git
cd diffusion-language-model
pip install -r requirements.txt
```
Pretrained checkpoints Huggingface [Repo](https://huggingface.co/Govind222/Diffusion-Language-Model-Base)

### 2. Requirements
Refer to this [page](https://pytorch.org/get-started/locally/) for Torch with cuda installation 
```bash
transformers
tokenizers
streamlit
numpy
tqdm
matplotlib
```

### 3. Using Pre-trained Model (Recommended)

Download the pre-trained model From [here](https://huggingface.co/Govind222/Diffusion-Language-Model-Base) and run the interactive demo:

```bash
cd Pre-Training
streamlit run ui.py
```

This launches an interactive web interface where you can:
- Input text prompts
- Watch the diffusion process in real-time
- Adjust generation parameters
- Compare with traditional autoregressive generation

### 4. Training from Scratch

If you want to train your own model:

```bash
# Prepare your data
# Place your training text in data/data.txt

# Start training
python train.py
```

#### Training Configuration:

The training script includes a centralized configuration system. Key parameters:

```python
CONFIG = {
    'vocab_size': 12000,
    'd_model': 320,
    'n_heads': 8,
    'n_layers': 12,
    'batch_size': 256,
    'learning_rate': 3e-5,
    'max_len': 256,
    'timesteps': 50
}
```

## 🎛️ Model Configuration

### Architecture Tuning:
- **d_model**: Embedding dimension (default: 320)
- **n_layers**: Number of transformer layers (default: 12)
- **n_heads**: Number of attention heads (default: 8)
- **d_ff**: Feed-forward dimension (default: 1280)

### Training Tuning:
- **batch_size**: Training batch size (default: 256)
- **learning_rate**: AdamW learning rate (default: 3e-5)
- **timesteps**: Diffusion denoising steps (default: 50)
- **max_len**: Maximum sequence length (default: 256)

### Diffusion Tuning:
- **noise_schedule**: Cosine beta scheduling
- **masking_strategy**: Token-level masking for text
- **denoising_steps**: Iterative refinement steps

## 📊 Performance & Benchmarks

### Training Metrics:
- **Dataset**: 110+ OCR-scanned Marathi agricultural textbooks
- **Training Time**: ~8 hours on NVIDIA L4
- **Memory Usage**: ~12GB GPU memory
- **Final Loss**: <2.5 (cross-entropy)

### Inference Speed:
- **Traditional Autoregressive**: ~0.5 tokens/second (sequential)
- **Our Diffusion Model**: ~25 tokens/second (parallel, 50 steps)
- **Speed Improvement**: 5-10x faster for comparable quality

## 🔬 Research Contributions

1. **Novel Time Conditioning**: Custom sinusoidal embeddings for diffusion timesteps
2. **Memory Optimization**: Efficient training for resource-constrained environments
3. **Real-time Visualization**: Interactive demonstration of diffusion process
4. **Regional Language Support**: Optimized tokenization for Marathi text
5. **Complete ML Pipeline**: Research → training → deployment → demo

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution:
- Scaling to larger model sizes
- Additional language support
- Inference optimization
- Novel diffusion schedules
- Evaluation metrics

## 📚 Technical Details

### Diffusion Process:
1. **Forward Process**: Gradually mask tokens according to cosine schedule
2. **Reverse Process**: Iteratively denoise masked tokens
3. **Conditioning**: Time embeddings guide denoising intensity
4. **Training**: Predict original tokens from noised versions

### Key Algorithms:
- **Cosine Beta Scheduling**: Smooth noise addition
- **Time Embedding**: Sinusoidal positional encoding for timesteps
- **Gradient Checkpointing**: Memory-efficient backpropagation
- **Mixed Precision**: FP16 training for efficiency

## 🚦 Roadmap

- [ ] **Scale to 100M+ parameters**
- [ ] **Multi-language support** (Hindi, English, Marathi etc.)
- [ ] **Production optimization** (faster inference)
- [ ] **Research publication** on visualization techniques
- [ ] **Advanced controllability** features
- [ ] **Benchmark comparisons** with GPT models
- [ ] **Mobile deployment** optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LLDM Paper**: [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
- **Dataset**: Custom OCR from 110+ Marathi agricultural textbooks
- **Inspiration**: Large Language Diffusion Models research
- **Community**: Open source ML research community

## 📞 Contact

- **Author**: [Govind Barbade]
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/govind-barbade-4a09b9251/)
- **Email**: [govindbarbade5@gmail.com]

## 🔗 References

1. [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
2. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)

---

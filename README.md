# Cross-Model Feature Dictionary (SAE Alignment) - Research Project

**Status: Research Complete - Available for Takeover**

This repository contains a completed research project on cross-model feature alignment using Sparse Autoencoders (SAEs) and Hypernetworks. The project successfully demonstrated that different language models can learn related internal representations through non-linear mappings, achieving significant improvements over traditional linear alignment methods.

## 🎯 **Key Findings**

### **Hypernetwork Approach vs Direct Alignment**
- **Direct Procrustes Alignment**: 0.7% cosine similarity (essentially no alignment)
- **Hypernetwork Mapping**: 60-80% cosine similarity (100x improvement!)

### **Bidirectional Transfer Success**
- **Pythia-410m → GPT-2 Medium**: 79.1% cosine similarity
- **GPT-2 Medium → Pythia-410m**: 75.7% cosine similarity
- **Feature Correlation**: 1,800-2,400 correlated features identified

## 📋 **Methodology**

### **Phase 1: Sparse Autoencoder Training**
1. **Extract Activations**: Collect hidden activations from both models on the same text corpus
2. **Train SAEs**: Learn sparse feature dictionaries for each model's activations
3. **Feature Extraction**: Encode activations into sparse, interpretable features

### **Phase 2: Cross-Model Alignment**
1. **Direct Alignment** (Baseline): Use Orthogonal Procrustes to find linear transformations
2. **Hypernetwork Approach** (Novel): Train neural networks to map between feature spaces

### **Phase 3: Feature Transfer & Validation**
1. **Bidirectional Transfer**: Test A→B and B→A feature mapping
2. **Causal Interventions**: Inject transferred features and observe behavioral changes
3. **Quality Metrics**: Cosine similarity, correlation analysis, sparsity preservation

## 🔬 **Technical Implementation**

### **Architecture**
- **Models**: Pythia-410m-deduped-v0 and GPT-2 Medium
- **Layers**: 5, 6, 7 (resid_pre activations)
- **SAE Configuration**: 4096-dimensional sparse features, L1 regularization
- **Hypernetwork**: 3-layer MLP with LayerNorm, Dropout, and multiple loss components

### **Key Innovations**
1. **Multi-Objective Loss**: Combines MSE, cosine similarity, correlation preservation, and sparsity matching
2. **Bidirectional Training**: Separate hypernetworks for each direction
3. **Robust Evaluation**: Multiple metrics including R², feature-wise correlation, and sparsity similarity

## 📊 **Results Summary**

| Metric | Direct Alignment | Hypernetwork |
|--------|------------------|--------------|
| Cosine Similarity | 0.7% | 60-80% |
| Matched Features | 0 | 1,800-2,400 |
| Transfer Success | ❌ | ✅ |
| Bidirectional | ❌ | ✅ |

### **Layer-wise Performance**
- **Layer 5**: 67.4% A→B, 58.9% B→A
- **Layer 6**: 79.1% A→B, 75.7% B→A  
- **Layer 7**: Results available in saved files

## 🚀 **Getting Started**

### **Prerequisites**
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### **Quick Start**
```bash
# 1. Train SAEs (already completed, saved in runs/demo/)
python scripts/train_saes.py --config configs/default.yaml

# 2. Train hypernetworks (already completed)
python scripts/train_hypernetworks.py --config configs/default.yaml --layers 6

# 3. Test feature transfer
python scripts/test_hypernetwork_transfer.py --config configs/default.yaml --layer 6
```

### **Configuration**
- **Small Scale**: 100 documents, 128 max length (for testing)
- **Full Scale**: 2000 documents, 256 max length (for research)
- **Models**: Configurable in `configs/default.yaml`

## 📁 **Project Structure**

```
Cross-Model-SAE/
├── src/
│   ├── sae.py              # Sparse Autoencoder implementation
│   ├── hypernet.py         # Hypernetwork for cross-model mapping
│   ├── align.py            # Traditional alignment methods
│   ├── interventions.py    # Feature injection experiments
│   └── backend_tlens.py    # TransformerLens integration
├── scripts/
│   ├── train_saes.py       # SAE training pipeline
│   ├── train_hypernetworks.py  # Hypernetwork training
│   └── test_hypernetwork_transfer.py  # Transfer testing
├── runs/demo/              # Saved models and results
└── configs/                # Experiment configurations
```

## 🔍 **Research Implications**

### **Model Interpretability**
- **Shared Representations**: Different architectures learn related internal features
- **Non-linear Mappings**: Feature relationships are complex, not linear
- **Transferable Knowledge**: Features can be meaningfully transferred between models

### **Practical Applications**
- **Knowledge Distillation**: Transfer learned features between model families
- **Model Comparison**: Quantify similarities between different architectures
- **Feature Analysis**: Understand what different models learn
- **Cross-Model Interventions**: Manipulate one model using another's features

## 🎯 **Future Research Directions**

### **Immediate Opportunities**
1. **Scale Up**: Test with larger models (GPT-3, Llama, etc.)
2. **More Architectures**: Compare across different model families
3. **Feature Analysis**: Analyze what specific features transfer best
4. **Downstream Tasks**: Test transfer on specific NLP tasks

### **Advanced Extensions**
1. **Multi-Model Alignment**: Align features across 3+ models simultaneously
2. **Hierarchical Transfer**: Transfer features at multiple layers
3. **Task-Specific Alignment**: Align features for specific downstream tasks
4. **Interpretable Mappings**: Understand what the hypernetworks learn

## 📈 **Performance Optimization**

### **Current Bottlenecks**
- **Memory**: Large SAEs (4096 dim) require significant GPU memory
- **Training Time**: Hypernetwork training takes 1-2 hours per layer
- **Data Requirements**: Need substantial text corpus for stable training

### **Optimization Strategies**
- **Reduced SAE Width**: Try 1024 or 2048 dimensional features
- **Gradient Accumulation**: For memory-constrained setups
- **Mixed Precision**: Use FP16 for faster training
- **Data Efficiency**: Explore few-shot alignment methods

## 🤝 **Contributing**

This project is **open for takeover**! The codebase is well-structured and documented. Key areas for contribution:

1. **Scale Experiments**: Test with larger models and datasets
2. **New Architectures**: Implement different alignment methods
3. **Analysis Tools**: Build better visualization and analysis tools
4. **Applications**: Apply to specific NLP tasks or model families

## 📚 **References**

- **Sparse Autoencoders**: [Anthropic's SAE work](https://transformer-circuits.pub/2023/monosemantic-features/)
- **Cross-Model Alignment**: [Procrustes analysis](https://en.wikipedia.org/wiki/Procrustes_analysis)
- **Hypernetworks**: [Ha et al. (2016)](https://arxiv.org/abs/1609.09106)
- **TransformerLens**: [Nanda & Bloom (2022)](https://github.com/neelnanda-io/TransformerLens)

## 📄 **License**

MIT License - feel free to use, modify, and distribute this research code.

---

**Note**: This project demonstrates that cross-model feature alignment is possible and meaningful, opening new avenues for model interpretability and knowledge transfer research. The hypernetwork approach significantly outperforms traditional linear methods, suggesting that model representations are more related than previously thought.

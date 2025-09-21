# LLM SLED Comparison Tool

A comprehensive Python implementation for comparing standard LLM decoding methods with SLED (Sliding Window LLM Decoding) optimization technique for memory-efficient inference.

## 🔍 Overview

This project provides a direct comparison between standard autoregressive language model decoding and SLED (Sliding Window LLM Decoding), an optimization technique that enables memory-efficient inference by using a sliding window approach. SLED is particularly useful for generating longer sequences while maintaining reasonable memory usage.

## ✨ Features

- **Dual Decoder Implementation**: Both standard autoregressive and SLED decoders
- **Performance Benchmarking**: Comprehensive time and memory usage comparison
- **Visualization Tools**: Automatic generation of comparison charts and graphs
- **Configurable Parameters**: Adjustable window sizes, stride lengths, and generation settings
- **Multiple Model Support**: Compatible with any Hugging Face transformer model
- **Statistical Analysis**: Multiple runs for reliable performance metrics

## 🚀 Quick Start

### Installation

#### Option 1: Install from source
```bash
# Clone the repository
git clone https://github.com/slkreddy/llm-sled-comparison.git
cd llm-sled-comparison

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

#### Option 2: Install core dependencies only
```bash
pip install torch>=2.0.0 transformers>=4.30.0 numpy>=1.21.0 matplotlib>=3.4.0 seaborn>=0.11.0 pandas>=1.3.0 psutil>=5.8.0
```

### Basic Usage

#### Run the comparison tool
```bash
# Using the command-line interface
sled-comparison

# Or run directly
python sled_comparison.py
```

#### Programmatic Usage
```python
from sled_comparison import ComparisonRunner

# Initialize the comparison runner
runner = ComparisonRunner(model_name="gpt2")

# Define test prompts
prompts = [
    "The future of artificial intelligence is",
    "Climate change represents a challenge that",
    "Advances in machine learning have"
]

# Run comparison
results = runner.run_comparison(
    prompts=prompts,
    max_length=100,
    temperature=0.8,
    num_runs=3
)

# Analyze and visualize results
runner.analyze_results()
```

## 📊 Performance Comparison

The tool provides detailed metrics comparing both approaches:

- **Generation Time**: Time taken to generate sequences
- **Memory Usage**: Peak memory consumption during generation
- **Throughput**: Tokens generated per second
- **Memory Efficiency**: Memory usage reduction percentage

### Expected Performance Characteristics

| Metric | Standard Decoding | SLED Decoding | Improvement |
|--------|------------------|---------------|-------------|
| Memory Usage | Higher | Lower | ~20-40% reduction |
| Generation Speed | Baseline | Slightly slower | Depends on window size |
| Long Sequences | Memory grows linearly | Constant memory | Significant for long texts |

## ⚙️ Configuration

### SLED Parameters

- **`window_size`**: Size of the sliding window (default: 50)
- **`stride`**: Step size for window movement (default: 25)
- **`model_name`**: Hugging Face model identifier (default: "gpt2")

### Generation Parameters

- **`max_length`**: Maximum sequence length to generate
- **`temperature`**: Sampling temperature for generation
- **`num_runs`**: Number of runs for statistical significance

### Example Configuration
```python
# Custom SLED decoder
sled_decoder = SLEDDecoder(
    model_name="gpt2-medium",
    window_size=100,
    stride=50
)

# Custom generation parameters
results = runner.run_comparison(
    prompts=prompts,
    max_length=200,
    temperature=1.0,
    num_runs=5
)
```

## 🔧 Advanced Usage

### Using Different Models
```python
# Use different model sizes
runner_small = ComparisonRunner(model_name="gpt2")
runner_medium = ComparisonRunner(model_name="gpt2-medium")
runner_large = ComparisonRunner(model_name="gpt2-large")
```

### Custom Window Configurations
```python
# Test different window sizes
for window_size in [25, 50, 100, 200]:
    sled_decoder = SLEDDecoder(
        model_name="gpt2",
        window_size=window_size,
        stride=window_size // 2
    )
    # Run comparison...
```

### Batch Processing
```python
# Process multiple prompt sets
prompt_sets = [
    ["Tech prompts..."],
    ["Science prompts..."],
    ["Literature prompts..."]
]

for i, prompts in enumerate(prompt_sets):
    results = runner.run_comparison(prompts=prompts)
    # Save results for each set
```

## 📁 Project Structure

```
llm-sled-comparison/
├── sled_comparison.py      # Main comparison implementation
├── requirements.txt        # Project dependencies
├── setup.py               # Package installation script
├── README.md              # This file
├── .gitignore             # Git ignore patterns
└── examples/              # Usage examples (if added)
```

## 🛠️ Development

### Setting up Development Environment
```bash
# Clone repository
git clone https://github.com/slkreddy/llm-sled-comparison.git
cd llm-sled-comparison

# Install development dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code
black sled_comparison.py

# Lint code
flake8 sled_comparison.py
```

### Optional Development Tools
```bash
# Install all optional dependencies
pip install -e ".[all]"

# Or install specific groups
pip install -e ".[notebooks]"  # Jupyter notebook support
pip install -e ".[plotting]"   # Advanced plotting with Plotly
pip install -e ".[serving]"    # API serving capabilities
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

### Core Dependencies
- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Pandas >= 1.3.0
- psutil >= 5.8.0

### Optional Dependencies
- pytest >= 7.0.0 (testing)
- black >= 22.0.0 (code formatting)
- flake8 >= 4.0.0 (linting)
- jupyter >= 1.0.0 (notebooks)
- plotly >= 5.0.0 (advanced plotting)
- fastapi >= 0.95.0 (API serving)

## 🐛 Troubleshooting

### Common Issues

#### Memory Errors
- Reduce `max_length` parameter
- Use smaller model (e.g., "gpt2" instead of "gpt2-large")
- Reduce `window_size` in SLED decoder

#### Slow Performance
- Use GPU if available: `torch.cuda.is_available()`
- Reduce `num_runs` for faster testing
- Use smaller models for initial testing

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Or install missing packages individually
pip install torch transformers matplotlib seaborn pandas psutil
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face team for the excellent `transformers` library
- PyTorch team for the deep learning framework
- The research community for SLED and related optimization techniques

## 📞 Contact

- **Author**: LaxmiKumar Reddy Sammeta
- **Email**: slkreddysite@gmail.com
- **GitHub**: [@slkreddy](https://github.com/slkreddy)
- **Issues**: [GitHub Issues](https://github.com/slkreddy/llm-sled-comparison/issues)

## 🔗 Related Work

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SLED: Memory-Efficient LLM Inference]()

---

**Note**: This implementation is for research and educational purposes. For production use, consider additional optimizations and error handling.

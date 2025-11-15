# Local LLM Fine-tuning with Ollama

This repository contains notebooks and resources for fine-tuning large language models (LLMs) locally and deploying them using Ollama. The workflow includes fine-tuning models using Unsloth, converting them to GGUF format, and running inference both with llama.cpp and Ollama.

## 📁 Repository Structure

```
local-llm-ollama/
├── llama_cpp_inference_chat.ipynb    # Notebook for running inference with llama-cpp-python
├── add_model_to_ollama.md            # Guide for adding GGUF models to Ollama
├── Meta-Llama-3.1-8B-q4_k_m-paul-graham-guide-GGUF/  # Example fine-tuned model
│   ├── Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf        # GGUF model file
│   ├── Modelfile                                     # Ollama Modelfile
│   └── README.md                                     # Model-specific documentation
└── README.md                          # This file
```

## 🚀 Features

- **Fine-tuning**: Fine-tune LLMs using Unsloth for efficient training
- **Model Conversion**: Convert fine-tuned models to GGUF format for efficient inference
- **Local Inference**: Run models locally using llama-cpp-python
- **Ollama Integration**: Deploy models to Ollama for easy API access
- **Notebook-based Workflow**: Jupyter notebooks for interactive development

## 📋 Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**
- **Ollama** (for model deployment)
- **CUDA-capable GPU** (recommended for fine-tuning)

### Python Packages

Install the required Python packages:

```bash
pip install llama-cpp-python
pip install jupyter
pip install unsloth  # For fine-tuning
```

For GPU support with llama-cpp-python:

```bash
# For CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

## 🎯 Quick Start

### 1. Running Inference with llama-cpp-python

Open `llama_cpp_inference_chat.ipynb` and follow the notebook to:

- Load a GGUF model
- Create chat completions
- Interact with your fine-tuned model

Example usage:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/your/model.gguf"
)

response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Your question here"}
    ]
)
```

### 2. Adding Models to Ollama

Follow the guide in `add_model_to_ollama.md` to:

1. Create a Modelfile with the appropriate template
2. Import your GGUF model into Ollama
3. Run inference via Ollama's API

Quick steps:

```bash
# Create Modelfile (see add_model_to_ollama.md for template)
nano Modelfile

# Create model in Ollama
ollama create my-model -f Modelfile

# Run inference
ollama run my-model "Your prompt here"
```

## 🔧 Fine-tuning Workflow

### Step 1: Fine-tune with Unsloth

Use Unsloth to fine-tune your base model:

```python
from unsloth import FastLanguageModel

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Fine-tune your model
# ... (your fine-tuning code)
```

### Step 2: Convert to GGUF

After fine-tuning, convert your model to GGUF format:

```python
# Save model
model.save_pretrained("my-finetuned-model")

# Convert to GGUF using llama.cpp tools
# This step typically involves using llama.cpp's conversion scripts
```

### Step 3: Deploy

- Use `llama_cpp_inference_chat.ipynb` for direct inference
- Or add to Ollama using `add_model_to_ollama.md` guide

## 📚 Example Model

This repository includes an example fine-tuned model:
- **Model**: Meta-Llama-3.1-8B-Instruct (fine-tuned on Paul Graham essays)
- **Format**: GGUF (Q4_K_M quantization)
- **Location**: `Meta-Llama-3.1-8B-q4_k_m-paul-graham-guide-GGUF/`

## 🔗 Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Modelfile Reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

## 📝 Notes

- GGUF files are gitignored by default (see `.gitignore`)
- Model files can be large; consider using Git LFS or external storage
- Different models require different chat templates; check the model's documentation
- Quantization levels (Q4_K_M, Q8_0, etc.) affect model size and quality

## 🤝 Contributing

Feel free to add your fine-tuning notebooks and share your models! When adding new models:

1. Create a directory with a descriptive name
2. Include the GGUF file and Modelfile
3. Add a README.md with model details and usage instructions

## 📄 License

Please check the license of the base models you use. Fine-tuned models inherit the license of their base models.

---

**Happy Fine-tuning! 🎉**


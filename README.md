# Multi-Modal Civil Complaint Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research project focused on classifying civil complaints using multi-modal data (Images + Text). This project utilizes **KoCLIP** to perform zero-shot and few-shot classification across 44 distinct categories of civil issues, ranging from road damage to public facility maintenance.

## 🚀 Key Features

- **Multi-Modal Support**: Leverages both visual and textual information for higher accuracy in complaint classification.
- **KoCLIP Integration**: Utilizes the `Bingsu/clip-vit-base-patch32-ko` model, optimized for the Korean language and visual-text alignment.
- **Zero-Shot Classification**: Capable of classifying complaints without extensive fine-tuning by leveraging the semantic relationship between complaint descriptions and input data.
- **Flexible Input**: Supports Image-only, Text-only, or combined Multi-modal inputs.
- **Extensive Taxonomy**: Handles 44 refined categories covering transportation, environment, public safety, and infrastructure.

## 📁 Project Structure

```text
study-multi-modal-civil-complaint-classifier/
├── src/
│   ├── models/            # Core model definitions (CLIP-based classifier)
│   ├── data/              # Data loading and preprocessing pipelines
│   ├── training/          # Training and fine-tuning logic
│   ├── evaluation/        # Metrics and model performance analysis
│   └── utils/             # Helper functions and logging
├── notebooks/              # Experimental research and EDA
├── data/                   # Dataset storage (Raw/Processed/External)
├── configs/                # Configuration files (YAML)
├── scripts/                # Standalone execution scripts (Train/Eval)
├── models/                 # Checkpoints and saved models
├── experiments/            # Detailed logs and experimental results
└── requirements.txt        # Dependency specification
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (optional but recommended for faster inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/pileuszu/study-multi-modal-civil-complaint-classifier.git
cd study-multi-modal-civil-complaint-classifier

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### 1. Research & Exploration
Explore the core logic and run sample predictions using the provided Jupyter notebook:
```bash
jupyter notebook notebooks/simple_civil_complaint_classifier.ipynb
```

### 2. Model Usage
You can use the `CLIPComplaintClassifier` directly in your code:

```python
from src.models.clip_classifier import CLIPComplaintClassifier
from PIL import Image

# Initialize classifier
classifier = CLIPComplaintClassifier()

# Define labels (or use defaults)
labels = ["Road damage (pothole)", "Illegal trash dumping", "Broken streetlight"]

# Predict
results = classifier.predict_top_k(
    labels=labels,
    text="There is a large hole in the middle of the road.",
    image=Image.open("path/to/image.jpg"),
    k=3
)

for label, prob in results:
    print(f"{label}: {prob:.4f}")
```

### 3. Training & Evaluation (Work in Progress)
Execute the training pipeline using configuration files:
```bash
python scripts/train.py --config configs/default.yaml
```

## 📊 Model Details

- **Base Model**: [KoCLIP (Bingsu/clip-vit-base-patch32-ko)](https://huggingface.co/Bingsu/clip-vit-base-patch32-ko)
- **Temperature**: 0.07 (default)
- **Classification Method**: Semantic similarity between input embeddings and label embeddings.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



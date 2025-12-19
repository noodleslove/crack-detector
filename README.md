# Crack Detector: AI-Powered Concrete Crack Detection System

A deep learning-based web application that detects cracks in concrete structures using transfer learning with ResNet18. This project demonstrates end-to-end machine learning engineering, from problem formulation to deployment.

## Table of Contents

- [Overview](#overview)
- [Design Thinking Process](#design-thinking-process)
- [Technical Architecture](#technical-architecture)
- [Model Training Pipeline](#model-training-pipeline)
- [Getting Started](#getting-started)
- [Streamlit Application](#streamlit-application)
- [FastAPI Backend](#fastapi-backend)
- [Results and Performance](#results-and-performance)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

## Overview

Crack detection is vital for structural health monitoring and inspection of concrete infrastructure. This project implements a computer vision solution to automatically classify images as containing cracks (positive) or no cracks (negative), achieving **99.81% accuracy** on validation data.

### Key Features

- Deep learning-based crack detection using ResNet18 transfer learning
- Interactive Streamlit web interface for real-time predictions
- FastAPI backend for scalable deployments
- Comprehensive Jupyter notebooks documenting the training pipeline
- Pre-trained model ready for inference

## Design Thinking Process

### 1. Empathize: Understanding the Problem

**Problem Statement**: Manual inspection of concrete structures is time-consuming, costly, and prone to human error. Infrastructure maintenance teams need an automated, reliable way to detect structural cracks early.

**User Needs**:

- Fast, accurate crack detection from images
- Easy-to-use interface requiring no technical expertise
- Confidence scores for predictions
- Support for various image formats and sizes

### 2. Define: Problem Scope

**Goal**: Build a binary image classifier that distinguishes between cracked and non-cracked concrete surfaces.

**Success Metrics**:

- Classification accuracy > 95%
- Inference time < 2 seconds per image
- User-friendly interface
- Model size < 100MB for easy deployment

### 3. Ideate: Solution Approach

**Chosen Approach**: Transfer learning with pre-trained ResNet18

**Why Transfer Learning?**

- Leverage features learned from ImageNet (1M+ images)
- Faster training with limited data
- Better generalization than training from scratch
- Proven architecture for image classification

**Alternative Approaches Considered**:

- Linear classifier (baseline): Simple but limited accuracy (~60%)
- Custom CNN from scratch: Requires more data and training time
- Other architectures (VGG, EfficientNet): ResNet18 provides optimal balance

### 4. Prototype: Implementation

The solution was built iteratively:

1. Data exploration and preprocessing
2. Baseline linear classifier
3. Transfer learning with ResNet18
4. Web application development

### 5. Test: Validation

- Achieved 99.81% accuracy on 10,000 validation samples
- Tested with various image conditions (lighting, angles, quality)
- User interface validated for ease of use

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌──────────────────┐           ┌──────────────────┐       │
│  │   Streamlit UI   │           │   FastAPI REST   │       │
│  │   (app/main.py)  │           │ (backend/main.py)│       │
│  └────────┬─────────┘           └────────┬─────────┘       │
└───────────┼──────────────────────────────┼─────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Model Layer                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  ResNet18 Transfer Learning Model (model.pt)       │     │
│  │  - Pre-trained on ImageNet                         │     │
│  │  - Modified final layer for binary classification  │     │
│  │  - Input: 224x224 RGB images                       │     │
│  │  - Output: [crack, no_crack] probabilities         │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────┐
│                    Data Layer                                │
│  - Concrete Crack Images Dataset (40,000 images)            │
│  - Training: 30,000 images (15K positive, 15K negative)     │
│  - Validation: 10,000 images (5K positive, 5K negative)     │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

- **Deep Learning**: PyTorch 2.5.1, TorchVision 0.20.1
- **Model Architecture**: ResNet18 (pre-trained on ImageNet)
- **Web Framework**: Streamlit 1.52.2 (UI), FastAPI 3.0.3 (API)
- **Image Processing**: PIL/Pillow 10.4.0
- **Python**: 3.12+
- **Package Management**: uv

## Model Training Pipeline

The complete training process is documented in Jupyter notebooks following a systematic approach:

### Notebook 1: Data Loading and Exploration

**File**: `notebooks/1.0_load_and_display_data.ipynb`

**Objectives**:

- Download and explore the concrete crack dataset
- Understand data distribution and characteristics
- Visualize sample images from both classes

**Key Steps**:

```python
# Dataset structure
├── Positive/  # 20,000 images with cracks
└── Negative/  # 20,000 images without cracks
```

**Learnings**:

- Dataset is balanced (50% positive, 50% negative)
- Images are 227x227 pixels, RGB color
- High-quality labeled images suitable for training

### Notebook 2: PyTorch Dataset Creation

**File**: `notebooks/2.1_data_loader_PyTorch.ipynb`

**Objectives**:

- Create custom PyTorch Dataset class
- Implement data loading and preprocessing
- Set up train/validation split

**Key Implementation**:

```python
class Dataset(Dataset):
    def __init__(self, transform=None, train=True):
        # Load positive and negative image paths
        # Interleave samples: [pos, neg, pos, neg, ...]
        # Split: 30,000 train / 10,000 validation

    def __getitem__(self, idx):
        # Load image, apply transforms
        # Return (image_tensor, label)
```

**Data Augmentation**:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])
```

### Notebook 3: Linear Classifier Baseline

**File**: `notebooks/3.1_linearclassiferPytorch.ipynb`

**Objectives**:

- Establish baseline performance with simple model
- Understand problem difficulty
- Compare against deep learning approach

**Model Architecture**:

```python
class LinearClassifier(nn.Module):
    def __init__(self, input_size=3*227*227, output_size=2):
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
```

**Training Configuration**:

- Optimizer: SGD (lr=0.1, momentum=0.1)
- Loss: CrossEntropyLoss
- Batch Size: 1000
- Epochs: 5

**Results**: ~60-70% accuracy (baseline)

### Notebook 4: ResNet18 Transfer Learning

**File**: `notebooks/4.1_resnet18_PyTorch.ipynb`

**Objectives**:

- Implement transfer learning with ResNet18
- Fine-tune for crack detection
- Achieve production-ready performance

**Transfer Learning Approach**:

```python
# Step 1: Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Step 2: Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace final layer for binary classification
model.fc = nn.Linear(512, 2)  # 512 inputs → 2 classes
```

**Why This Works**:

- ResNet18 learned rich visual features from ImageNet
- Early layers detect edges, textures, patterns (useful for cracks)
- Only final layer needs training for our specific task
- Significantly reduces training time and data requirements

**Training Configuration**:

- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch Size: 100
- Epochs: 1 (demonstration), can extend for better performance
- Training Samples: 30,000
- Validation Samples: 10,000

**Key Results**:

- **Validation Accuracy**: 99.81%
- **Training Time**: ~45 minutes (1 epoch)
- **Model Size**: ~45MB (model.pt)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- 2GB+ RAM for model inference
- (Optional) GPU for faster training

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/noodleslove/crack-detector.git
cd crack-detector
```

2. **Set up Python environment** (using uv - recommended):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Alternative**: Using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Verify installation**:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

## Streamlit Application

The Streamlit app provides an intuitive web interface for crack detection.

### Running the Streamlit App

```bash
# From project root directory
streamlit run app/main.py
```

The application will open automatically in your browser at `http://localhost:8501`

### Using the Application

1. **Upload an Image**:

   - Click "Browse files" in the sidebar
   - Select a JPG, JPEG, or PNG image (max 2MB)
   - Supported formats: .jpg, .jpeg, .png

2. **View Results**:

   - Uploaded image is displayed
   - Prediction shows: "Cracks detected" or "No cracks detected"
   - Real-time inference (< 2 seconds)

3. **Application Features**:
   - Clean, user-friendly interface
   - Image size validation (max 2MB)
   - Format validation
   - Instant visual feedback

### Application Code Overview

**File**: `app/main.py`

```python
# Load pre-trained model
model = torch.load("model.pt")
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(upload):
    img = Image.open(upload)
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)

    return "Cracks detected" if prediction.item() == 1 else "No cracks detected"
```

## FastAPI Backend

A production-ready REST API for integration into other applications.

### Running the FastAPI Server

```bash
# From project root directory
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at: `http://localhost:8000/docs`

### API Endpoints

#### POST /predict

Analyze an image for cracks.

**Request**:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

**Response**:

```json
{
  "output": [[2.14, -1.89]], // Raw logits
  "prediction": 1 // 1: crack, 0: no crack
}
```

**Features**:

- CORS enabled for cross-origin requests
- File validation (type, size)
- Secure filename handling
- Error handling with appropriate status codes

## Results and Performance

### Model Performance

| Metric                  | Value       |
| ----------------------- | ----------- |
| Validation Accuracy     | **99.81%**  |
| Training Samples        | 30,000      |
| Validation Samples      | 10,000      |
| Model Size              | 45 MB       |
| Inference Time          | < 2 seconds |
| Training Time (1 epoch) | ~45 minutes |

### Comparison with Baseline

| Model                          | Accuracy   | Parameters                  | Training Time |
| ------------------------------ | ---------- | --------------------------- | ------------- |
| Linear Classifier              | ~60-70%    | 309,002                     | 5 min         |
| **ResNet18 Transfer Learning** | **99.81%** | ~11M (only 1,026 trainable) | 45 min        |

### Key Insights

1. **Transfer Learning Impact**: Achieved 99.81% accuracy vs 60-70% with linear baseline
2. **Efficiency**: Only final layer trained (1,026 parameters) while leveraging 11M pre-trained parameters
3. **Generalization**: High validation accuracy indicates good generalization
4. **Production Ready**: Model performs well on unseen data with fast inference

## Project Structure

```
crack-detector/
├── app/
│   ├── main.py              # Streamlit web application
│   └── model.pt             # Trained ResNet18 model (45MB)
├── backend/
│   ├── main.py              # FastAPI REST API
│   └── model.pt             # Trained model (copy)
├── notebooks/
│   ├── 1.0_load_and_display_data.ipynb       # Data exploration
│   ├── 2.1_data_loader_PyTorch.ipynb         # Dataset creation
│   ├── 3.1_linearclassiferPytorch.ipynb      # Baseline model
│   └── 4.1_resnet18_PyTorch.ipynb            # Transfer learning
├── pyproject.toml           # Project dependencies
├── requirements.txt         # Frozen dependencies
└── README.md               # This file
```

## Future Enhancements

### Model Improvements

- [ ] Implement ensemble methods for higher accuracy
- [ ] Add crack severity classification (mild, moderate, severe)
- [ ] Detect and measure crack dimensions
- [ ] Support for multi-class crack types

### Application Features

- [ ] Batch image processing
- [ ] Historical analysis and tracking
- [ ] PDF report generation
- [ ] Mobile app development
- [ ] Real-time video stream analysis

### Technical Enhancements

- [ ] Model quantization for edge deployment
- [ ] ONNX export for cross-platform compatibility
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] A/B testing framework
- [ ] Model monitoring and retraining pipeline

## Acknowledgments

- **Dataset**: Concrete Crack Images for Classification (40,000 images)
- **Framework**: PyTorch and Streamlit communities
- **Architecture**: ResNet paper by He et al. (2015)
- **Transfer Learning**: Inspired by ImageNet pre-training approaches

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Project by**: Eddie | [GitHub](https://github.com/noodleslove) | [Portfolio](https://eddieho.xyz)

**Keywords**: Deep Learning, Computer Vision, Transfer Learning, PyTorch, ResNet18, Crack Detection, Structural Health Monitoring, Image Classification, Streamlit, FastAPI

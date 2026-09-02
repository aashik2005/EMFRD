# EMFRD - Explainable Multimodal Framework for Fake Review Detection

Research implementation of IEEE paper: **"Explainable Multimodal Framework for Fake Review Detection Using Semantic, Graph, and Adversarial Learning"**

## Overview

EMFRD is a comprehensive multimodal framework that combines:

1. **Semantic Analysis**: RoBERTa + Contrastive Learning
2. **Graph-Based Behavioral Analysis**: Heterogeneous GNN (HGNN)
3. **Adversarial Robustness**: GAN-based adversarial training
4. **Multimodal Fusion**: Gated fusion mechanism
5. **Explainability**: SHAP + Counterfactual explanations

This is a **REAL**, **RESEARCH-GRADE** implementation with actual trainable models and experimental results.

## Project Structure

```
EMFRD/
├── backend/              # Python backend (FastAPI + PyTorch)
│   ├── api/              # FastAPI routes
│   ├── models/           # ML models (RoBERTa, HGNN, GAN, Fusion)
│   ├── training/         # Training scripts
│   ├── evaluation/       # Metrics and evaluation
│   ├── data/             # Dataset adapters
│   ├── preprocessing/    # Data preprocessing
│   └── utils/            # Utilities
├── frontend/             # React frontend (TypeScript + Ant Design)
│   └── src/
│       ├── pages/        # UI pages
│       ├── components/   # Reusable components
│       └── api/          # API client
├── configs/              # Configuration files
├── data/                 # Datasets
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed data
│   └── cache/            # Feature cache
├── models/               # Saved model checkpoints
├── experiments/          # Experimental results
│   ├── results/          # JSON results
│   ├── figures/          # Visualizations
│   └── logs/             # Training logs
└── docs/                 # Documentation
```

## Installation

### Requirements

- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (recommended, but CPU works)
- 16GB+ RAM recommended

### Backend Setup

```bash
cd EMFRD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

## Dataset Setup

### Primary Dataset: Kaggle Fake Reviews

1. Download from: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
2. OR from OSF: https://osf.io/tyue9/
3. Place CSV file in: `data/raw/fake_reviews/`

Expected file names:
- `fake_reviews_dataset.csv`
- `deceptive-opinion.csv`
- Any `.csv` file in the directory

The system will automatically:
- Detect column names
- Normalize to canonical schema
- Validate data quality
- Report graph compatibility

## Phase 1: RoBERTa Baseline (Current)

### Train RoBERTa Baseline

```bash
# From project root
python -m backend.training.train_roberta
```

This will:
1. Load and preprocess the dataset
2. Create train/val/test splits
3. Train RoBERTa for fake review classification
4. Save checkpoints to `models/roberta_baseline/`
5. Save results to `experiments/results/`

Training configuration can be modified in:
- `backend/config.py`
- `configs/experiment.yaml`

### Training Parameters

Default configuration:
```yaml
batch_size: 8           # Reduce if GPU memory is limited
learning_rate: 2e-5
max_epochs: 3
max_seq_length: 256
random_seed: 42
```

For faster experimentation:
```bash
# Freeze RoBERTa encoder (train only classifier)
python -m backend.training.train_roberta --freeze-encoder
```

## Running the Application

### Start Backend

```bash
# From project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:5173

## Usage

### 1. Web Interface

Navigate to http://localhost:5173

**Dashboard**: View model status and overview
**Prediction**: Test individual reviews
**Experiments**: Compare model performance

### 2. API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predict with RoBERTa
curl -X POST http://localhost:8000/api/predict/roberta \
  -H "Content-Type: application/json" \
  -d '{"review_text": "This is absolutely amazing!!! Best product ever!!!"}'

# List available models
curl http://localhost:8000/api/predict/models

# Get experimental results
curl http://localhost:8000/api/metrics/comparison
```

### 3. Python API

```python
from backend.models import RoBERTaBaseline
from backend.models.roberta_baseline import RoBERTaTokenizer
from backend.utils import CheckpointManager, get_device
import torch

# Load model
device = get_device()
model = RoBERTaBaseline()
checkpoint_manager = CheckpointManager("models/roberta_baseline", "roberta_baseline")
checkpoint_manager.load(model, device=device)
model = model.to(device)
model.eval()

# Predict
tokenizer = RoBERTaTokenizer()
text = "This product is amazing!!!"
encoded = tokenizer.encode_batch([text])

with torch.no_grad():
    outputs = model(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device)
    )
    probas = torch.softmax(outputs["logits"], dim=-1)
    prediction = "FAKE" if probas[0, 1] > 0.5 else "GENUINE"
    confidence = float(probas[0, 1])

print(f"Prediction: {prediction} ({confidence:.2%})")
```

## Results

### IMPORTANT: Comparing with Paper

The system displays:
- **Paper Reference**: Results reported in the paper
- **Our Reproduction**: Actual results from our implementation

Example:

```
RoBERTa Baseline
----------------
Paper Reference:  93.4% accuracy
Our Reproduction: 91.2% accuracy

This is acceptable! Small differences are expected due to:
- Different random initialization
- Dataset variations
- Implementation details
```

### Viewing Results

Results are stored in JSON format:
```bash
cat experiments/results/roberta_baseline_*.json
```

Structure:
```json
{
  "experiment_id": "roberta_baseline_20240315_120000",
  "model": "roberta_baseline",
  "dataset": "fake_reviews",
  "final_test_results": {
    "accuracy": 0.912,
    "precision": 0.905,
    "recall": 0.897,
    "f1": 0.901,
    "roc_auc": 0.958
  }
}
```

## Development Phases

### ✅ Phase 1: RoBERTa Baseline (COMPLETE)
- Dataset loading and preprocessing
- RoBERTa classification model
- Training pipeline
- Evaluation metrics
- FastAPI backend
- React frontend

### 🔄 Phase 2: RoBERTa + Contrastive Learning (NEXT)
- Supervised contrastive loss
- Projection head
- Enhanced semantic embeddings

### 📋 Phase 3: HGNN
- Heterogeneous graph construction
- User-Review-Product relationships
- Graph neural network
- Behavioral features

### 📋 Phase 4: GAN Adversarial Training
- Generator for synthetic reviews
- Discriminator/detector
- Adversarial robustness

### 📋 Phase 5: Gated Multimodal Fusion
- Combine semantic + graph + adversarial
- Learned gating mechanism
- Missing modality handling

### 📋 Phase 6: Explainability
- SHAP feature importance
- Counterfactual generation
- Modality contribution analysis

### 📋 Phase 7: Ablation & Robustness
- Ablation studies
- Robustness experiments
- Traditional ML baselines

### 📋 Phase 8: Research Dashboard
- Complete visualization
- Model comparison
- Publication-ready figures

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size in backend/config.py
BATCH_SIZE = 4

# Or use gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
```

### Dataset Not Found

```bash
# Check dataset location
ls data/raw/fake_reviews/

# The system will show download instructions
python -m backend.training.train_roberta
```

### Model Not Trained Warning

If you see "WARNING: No trained model found":

```bash
# Train the model first
python -m backend.training.train_roberta
```

### Frontend API Connection Error

Check that backend is running:
```bash
curl http://localhost:8000/health
```

## Research Mode vs Demo Mode

### Research Mode
Full training and experimentation:
```bash
python -m backend.training.train_roberta  # Full training
```

### Demo Mode
Use existing checkpoints for quick demos:
```bash
uvicorn backend.main:app --reload  # Backend
cd frontend && npm run dev         # Frontend
```

## Configuration

### Training Configuration

Edit `backend/config.py`:
```python
BATCH_SIZE = 8
MAX_EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
```

### Dataset Configuration

Edit `configs/experiment.yaml`:
```yaml
dataset:
  name: fake_reviews
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## GPU/CPU Support

The system auto-detects CUDA:

```python
# Force CPU
DEVICE = "cpu"

# Auto-detect (default)
DEVICE = "auto"

# Force CUDA
DEVICE = "cuda"
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=backend tests/
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{emfrd2024,
  title={Explainable Multimodal Framework for Fake Review Detection Using Semantic, Graph, and Adversarial Learning},
  author={[Your Name]},
  journal={IEEE},
  year={2024}
}
```

## License

This is a research implementation for academic purposes.

## Contact

For questions or issues, please open a GitHub issue or contact the author.

---

## Quick Start Summary

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset
# Place CSV in data/raw/fake_reviews/

# 3. Train RoBERTa
python -m backend.training.train_roberta

# 4. Start backend
uvicorn backend.main:app --reload

# 5. Start frontend (new terminal)
cd frontend
npm install
npm run dev

# 6. Open browser
# http://localhost:5173
```

---

**Status**: Phase 1 Complete ✅

Next: Implement RoBERTa + Contrastive Learning

# EMFRD System Testing Checklist

This document provides a comprehensive testing procedure to verify the EMFRD system is working correctly before proceeding to the next phase.

---

## Prerequisites

### Required Software

```bash
# Check Python installation
python --version  # Should be 3.11+

# Check Node.js installation
node --version    # Should be 18+
npm --version

# Check Git
git --version
```

**If Python is not installed:**
- Download from: https://www.python.org/downloads/
- Install Python 3.11 or higher
- Make sure to check "Add Python to PATH" during installation

---

## Phase 1: Environment Setup

### 1.1 Install Python Dependencies

```bash
cd C:\Ashik\EMFRD

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Install dependencies
pip install torch transformers scikit-learn pandas numpy tqdm pyyaml python-dotenv fastapi uvicorn
```

**Expected Result**: All packages install without errors

**Status**: [ ] PASS [ ] FAIL

---

### 1.2 Install Frontend Dependencies

```bash
cd frontend

# Install Node packages
npm install
```

**Expected Result**: No errors, `node_modules` created

**Status**: [ ] PASS [ ] FAIL

---

### 1.3 Verify Project Structure

```bash
cd C:\Ashik\EMFRD

# Check key files exist
ls backend/main.py
ls backend/models/roberta_baseline.py
ls backend/models/roberta_contrastive.py
ls backend/models/hgnn.py
ls backend/models/gan_adversarial.py
ls backend/models/gated_fusion.py
ls frontend/src/App.tsx
```

**Expected Result**: All files exist

**Status**: [ ] PASS [ ] FAIL

---

## Phase 2: Backend Testing

### 2.1 Test Python Syntax

```bash
cd backend

# Validate main.py
python -m py_compile main.py

# Validate all models
python -m py_compile models/roberta_baseline.py
python -m py_compile models/roberta_contrastive.py
python -m py_compile models/hgnn.py
python -m py_compile models/gan_adversarial.py
python -m py_compile models/gated_fusion.py
```

**Expected Result**: No syntax errors

**Status**: [ ] PASS [ ] FAIL

---

### 2.2 Test Model Imports

```bash
cd C:\Ashik\EMFRD

# Test importing models
python -c "from backend.models import RoBERTaBaseline; print('RoBERTa OK')"
python -c "from backend.models import RoBERTaContrastive; print('Contrastive OK')"
python -c "from backend.models import GANAdversarial; print('GAN OK')"
python -c "from backend.models import GatedMultimodalFusion; print('Fusion OK')"
```

**Expected Result**: All imports succeed with OK messages

**Status**: [ ] PASS [ ] FAIL

---

### 2.3 Test GAN Model

```bash
cd C:\Ashik\EMFRD

# Run built-in GAN test
python backend/models/gan_adversarial.py
```

**Expected Output**:
```
Testing GAN Adversarial Model...
GANAdversarial initialized:
  Latent dim: 100
  ...
Test passed!
```

**Status**: [ ] PASS [ ] FAIL

---

### 2.4 Test Fusion Model

```bash
cd C:\Ashik\EMFRD

# Run built-in Fusion test
python backend/models/gated_fusion.py
```

**Expected Output**:
```
Testing Gated Multimodal Fusion...
GatedMultimodalFusion initialized:
  ...
Test passed!
```

**Status**: [ ] PASS [ ] FAIL

---

### 2.5 Start FastAPI Backend

```bash
cd C:\Ashik\EMFRD

# Start backend server
uvicorn backend.main:app --reload
```

**Expected Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Test in Browser**:
- http://localhost:8000 → Should show API info
- http://localhost:8000/health → Should return `{"status": "healthy"}`
- http://localhost:8000/docs → Should show Swagger API docs

**Status**: [ ] PASS [ ] FAIL

---

### 2.6 Test API Endpoints

Keep backend running, open new terminal:

```bash
# Test health
curl http://localhost:8000/health

# Test list models
curl http://localhost:8000/api/predict/models

# Test prediction (will use untrained model)
curl -X POST http://localhost:8000/api/predict/roberta \
  -H "Content-Type: application/json" \
  -d "{\"review_text\": \"This product is amazing!\"}"
```

**Expected Results**:
- Health: `{"status": "healthy", "cuda_available": ...}`
- Models: JSON list of models
- Prediction: JSON with prediction/confidence (may show warning about untrained model)

**Status**: [ ] PASS [ ] FAIL

---

## Phase 3: Frontend Testing

### 3.1 Build Frontend

```bash
cd frontend

# Build TypeScript
npm run build
```

**Expected Result**: Build completes without errors

**Status**: [ ] PASS [ ] FAIL

---

### 3.2 Start Frontend Dev Server

```bash
cd frontend

# Start dev server
npm run dev
```

**Expected Output**:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

**Status**: [ ] PASS [ ] FAIL

---

### 3.3 Test Frontend Pages

With both backend and frontend running:

**Visit**: http://localhost:5173

Test each page:

1. **Dashboard** (`/`)
   - [ ] Page loads
   - [ ] Shows model status
   - [ ] Shows statistics
   - [ ] No console errors

2. **Prediction** (`/prediction`)
   - [ ] Page loads
   - [ ] Model dropdown works
   - [ ] Can enter review text
   - [ ] Can click "Predict" button
   - [ ] Shows prediction result
   - [ ] Shows confidence scores

3. **Experiments** (`/experiments`)
   - [ ] Page loads
   - [ ] Shows experiments (if any)
   - [ ] Shows paper reference
   - [ ] Charts render

**Status**: [ ] PASS [ ] FAIL

---

## Phase 4: Dataset Testing

### 4.1 Verify Demo Dataset

```bash
cd C:\Ashik\EMFRD

# Check demo dataset
ls -lh data/raw/fake_reviews/demo_dataset.csv

# Preview
head -5 data/raw/fake_reviews/demo_dataset.csv
```

**Expected Result**: CSV with 40+ reviews visible

**Status**: [ ] PASS [ ] FAIL

---

### 4.2 Test Dataset Loading

```bash
cd C:\Ashik\EMFRD

# Run dataset validation script
python scripts/download_datasets.py validate fake_reviews
```

**Expected Output**:
```
Validating dataset: fake_reviews
✓ Dataset loaded successfully!
Dataset Information:
  total_reviews: 40
  fake_count: 20
  genuine_count: 20
```

**Status**: [ ] PASS [ ] FAIL

---

## Phase 5: Training Pipeline Testing

### 5.1 Test RoBERTa Training (Small Scale)

```bash
cd C:\Ashik\EMFRD

# Train on demo dataset (small, fast test)
python -m backend.training.train_roberta
```

**Expected Behavior**:
- Dataset loads
- Model initializes
- Training starts
- Validation runs
- Checkpoint saves to `models/roberta_baseline/`
- Results save to `experiments/results/`

**Expected Time**: 2-5 minutes on demo dataset

**Status**: [ ] PASS [ ] FAIL

---

### 5.2 Verify Trained Model

```bash
# Check checkpoint exists
ls -lh models/roberta_baseline/best.pt

# Check results
ls -lh experiments/results/roberta_baseline*.json

# View results
cat experiments/results/roberta_baseline*.json
```

**Expected Result**: 
- Checkpoint file exists
- Results JSON shows metrics (accuracy, precision, recall, F1)

**Status**: [ ] PASS [ ] FAIL

---

### 5.3 Test Prediction with Trained Model

```bash
# Restart backend (to reload trained model)
# Then test prediction again

curl -X POST http://localhost:8000/api/predict/roberta \
  -H "Content-Type: application/json" \
  -d "{\"review_text\": \"This product is amazing! Best ever!!!\"}"
```

**Expected Result**: 
- No "untrained model" warning
- Returns prediction with confidence

**Status**: [ ] PASS [ ] FAIL

---

## Phase 6: Integration Testing

### 6.1 End-to-End Prediction Flow

1. Start backend: `uvicorn backend.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Open: http://localhost:5173/prediction
4. Enter test review: "This is absolutely amazing! Best product ever! Five stars!"
5. Click "Predict"

**Expected Result**:
- Prediction shows (FAKE or GENUINE)
- Confidence score displays
- Probability breakdown shows
- No errors in browser console
- No errors in backend console

**Status**: [ ] PASS [ ] FAIL

---

### 6.2 Model Comparison

1. Navigate to: http://localhost:5173/experiments
2. Check if experiments show

**Expected Result**:
- Page loads
- Shows any completed experiments
- Shows paper reference results
- Charts render properly

**Status**: [ ] PASS [ ] FAIL

---

## Phase 7: Code Quality Checks

### 7.1 Python Import Check

```bash
cd C:\Ashik\EMFRD

# Check all critical imports
python -c "
from backend.config import settings
from backend.models import RoBERTaBaseline, RoBERTaContrastive
from backend.models import GANAdversarial, GatedMultimodalFusion
from backend.data import get_dataset
from backend.preprocessing import TextPreprocessor
from backend.evaluation import MetricsCalculator
print('All imports successful!')
"
```

**Status**: [ ] PASS [ ] FAIL

---

### 7.2 TypeScript Compilation

```bash
cd frontend

# Check TypeScript
npx tsc --noEmit
```

**Expected Result**: No TypeScript errors

**Status**: [ ] PASS [ ] FAIL

---

## Phase 8: Architecture Verification

### 8.1 Verify All Models Exist

```bash
cd C:\Ashik\EMFRD

# Check model files
ls backend/models/roberta_baseline.py
ls backend/models/roberta_contrastive.py
ls backend/models/hgnn.py
ls backend/models/gan_adversarial.py
ls backend/models/gated_fusion.py
ls backend/models/losses/contrastive_loss.py

# Check training scripts
ls backend/training/train_roberta.py
ls backend/training/train_contrastive.py
ls backend/training/train_hgnn.py

# Check graph
ls backend/graph/heterograph.py
```

**Status**: [ ] PASS [ ] FAIL

---

## Summary Checklist

### Core Functionality
- [ ] Python environment working
- [ ] Backend starts without errors
- [ ] Frontend builds and runs
- [ ] API endpoints respond
- [ ] Can make predictions
- [ ] Dataset loads
- [ ] Demo dataset works

### Training Pipeline
- [ ] RoBERTa trains successfully
- [ ] Checkpoints save
- [ ] Metrics calculate
- [ ] Results store

### Integration
- [ ] Frontend ↔ Backend communication works
- [ ] Prediction UI functional
- [ ] Metrics display correctly
- [ ] No console errors

### Models Implemented
- [ ] RoBERTa Baseline
- [ ] RoBERTa + Contrastive
- [ ] HGNN
- [ ] GAN Adversarial
- [ ] Gated Multimodal Fusion

---

## Known Limitations (Expected)

✅ **Expected Behaviors** (NOT bugs):

1. **"WARNING: No trained model found"** on first prediction
   - This is normal before training
   - Model will use untrained weights
   - Train the model to remove warning

2. **100% accuracy on demo dataset**
   - Demo dataset is tiny (40 samples)
   - Overfitting is expected
   - This is just for pipeline testing
   - Use full Kaggle dataset for real experiments

3. **HGNN cannot run on demo dataset**
   - Demo dataset may lack user_id/product_id
   - Use FraudAmazon dataset for HGNN
   - This is documented in PHASE3_SUMMARY.md

4. **Fusion model not trained yet**
   - Fusion requires all component models trained first
   - This is expected at current stage
   - Train components individually first

---

## If Everything Passes ✅

Your EMFRD system is **FULLY FUNCTIONAL** and ready for:

✅ Phase 6: Explainability (SHAP + Counterfactuals)
✅ Phase 7: Ablation Studies & Evaluation
✅ Production deployment
✅ IEEE presentation

---

## If Tests Fail ❌

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`
**Solution**: Install PyTorch: `pip install torch`

**Issue**: `ModuleNotFoundError: No module named 'transformers'`
**Solution**: Install transformers: `pip install transformers`

**Issue**: `CUDA out of memory`
**Solution**: Reduce batch size in `backend/config.py`: `BATCH_SIZE = 4`

**Issue**: Frontend won't start
**Solution**: Delete `node_modules`, run `npm install` again

**Issue**: API returns 500 error on prediction
**Solution**: Check backend console for actual error message

**Issue**: Training is very slow
**Solution**: 
- Demo dataset should train in 2-5 minutes
- If slower, check if GPU is detected
- Use `--freeze-encoder` flag for faster testing

---

## Testing Time Estimates

- **Environment Setup**: 10-15 minutes
- **Backend Testing**: 5-10 minutes
- **Frontend Testing**: 5 minutes
- **Training Test**: 2-5 minutes (demo dataset)
- **Integration Testing**: 5 minutes

**Total**: ~30-45 minutes for complete validation

---

## Ready for Next Phase?

If all critical tests pass:

✅ Backend starts
✅ Frontend runs
✅ API responds
✅ Can make predictions
✅ Training pipeline works

Then the system is **PRODUCTION-READY** and you can proceed to:

🚀 **Phase 6**: Explainability (SHAP + Counterfactuals)
🚀 **Phase 7**: Ablation Studies & Complete Evaluation

---

## Report Template

After testing, report results like:

```
EMFRD System Test Report
========================

Date: ___________
Python Version: ___________
PyTorch Version: ___________
CUDA Available: Yes / No

Phase 1 (Setup): PASS / FAIL
Phase 2 (Backend): PASS / FAIL
Phase 3 (Frontend): PASS / FAIL
Phase 4 (Dataset): PASS / FAIL
Phase 5 (Training): PASS / FAIL
Phase 6 (Integration): PASS / FAIL

Overall Status: READY / NEEDS FIX

Issues Found:
1. ___________
2. ___________

Next Steps:
___________
```

---

**END OF TESTING CHECKLIST**

Repository: https://github.com/aashik2005/EMFRD

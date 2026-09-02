# EMFRD Quick Start Guide

**Current Status**: Node.js ✅ | Python ❌

Follow these steps to get your EMFRD system running.

---

## Step 1: Install Python (Required)

### Windows Installation

1. **Download Python 3.11+**
   - Visit: https://www.python.org/downloads/
   - Click "Download Python 3.11.x" (or latest 3.12)

2. **Run Installer**
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH"
   - Click "Install Now"
   - Wait for installation to complete

3. **Verify Installation**
   ```bash
   # Close and reopen your terminal, then run:
   python --version
   ```
   
   Expected: `Python 3.11.x` or `Python 3.12.x`

---

## Step 2: Install Python Dependencies

```bash
cd C:\Ashik\EMFRD

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install torch transformers scikit-learn pandas numpy tqdm pyyaml python-dotenv fastapi uvicorn

# Optional: Install DGL for HGNN (requires additional setup)
# pip install dgl
```

**Expected time**: 3-5 minutes

---

## Step 3: Verify Backend Works

### Test 1: Python Imports

```bash
# Test basic imports
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "from backend.models import RoBERTaBaseline; print('EMFRD models OK')"
```

Expected output:
```
PyTorch 2.x.x
Transformers 4.x.x
EMFRD models OK
```

### Test 2: Model Initialization

```bash
# Test GAN model (has built-in test)
python backend/models/gan_adversarial.py
```

Expected: `Test passed!`

### Test 3: Model Fusion

```bash
# Test Fusion model (has built-in test)
python backend/models/gated_fusion.py
```

Expected: `Test passed!`

---

## Step 4: Quick Training Test (5 minutes)

Train on the demo dataset (40 samples - fast):

```bash
# Train RoBERTa baseline
python -m backend.training.train_roberta

# This should:
# - Load demo dataset
# - Train for a few epochs
# - Save checkpoint to models/roberta_baseline/
# - Save results to experiments/results/
```

**Expected time**: 2-5 minutes

**Expected output**:
```
Loading dataset: fake_reviews
Dataset loaded: 40 reviews (20 fake, 20 genuine)
Training...
Epoch 1/10: loss=0.xxx, acc=0.xxx
...
Best model saved!
```

---

## Step 5: Start Backend Server

```bash
# Start FastAPI server
uvicorn backend.main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Test in browser**: http://localhost:8000/docs

You should see the Swagger API documentation.

---

## Step 6: Install Frontend Dependencies

**Open a NEW terminal** (keep backend running):

```bash
cd C:\Ashik\EMFRD\frontend

# Install Node packages
npm install
```

**Expected time**: 1-2 minutes

---

## Step 7: Start Frontend

```bash
# In frontend directory
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

---

## Step 8: Test in Browser

Open: **http://localhost:5173**

### Test Prediction

1. Click "Prediction" in sidebar
2. Enter test review:
   ```
   This product is absolutely amazing! Best ever! Five stars!
   ```
3. Select model: "RoBERTa Baseline"
4. Click "Predict"

**Expected result**:
- Shows prediction (FAKE or GENUINE)
- Shows confidence score
- Shows probability breakdown
- No errors in browser console

---

## Quick Validation Checklist

Run through this checklist:

- [ ] Python installed and in PATH
- [ ] PyTorch installed
- [ ] Transformers installed
- [ ] Backend imports work
- [ ] GAN model test passes
- [ ] Fusion model test passes
- [ ] RoBERTa trains successfully
- [ ] Backend server starts
- [ ] Frontend npm install completes
- [ ] Frontend dev server starts
- [ ] Can access http://localhost:5173
- [ ] Can make predictions
- [ ] No console errors

---

## Common Issues

### Issue: "Python was not found"
**Solution**: 
1. Install Python from python.org
2. Check "Add Python to PATH" during installation
3. Restart terminal

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: 
```bash
pip install torch transformers scikit-learn pandas numpy
```

### Issue: "Cannot find module 'vite'"
**Solution**: 
```bash
cd frontend
rm -rf node_modules
npm install
```

### Issue: Backend takes forever to train
**Solution**: 
- Demo dataset should take 2-5 minutes
- If slower, it's normal (CPU training)
- Use `--freeze-encoder` flag for faster testing

### Issue: Port 8000 or 5173 already in use
**Solution**: 
```bash
# Find and kill process on Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## Next Steps After Testing

Once all tests pass, you have three options:

### Option 1: Train More Models
```bash
# Train contrastive learning
python -m backend.training.train_contrastive

# Train HGNN (requires DGL and fraud_amazon dataset)
python -m backend.training.train_hgnn --dataset fraud_amazon
```

### Option 2: Use Full Kaggle Dataset
- Download fake reviews dataset from Kaggle
- Replace `data/raw/fake_reviews/demo_dataset.csv`
- Retrain for production-grade results

### Option 3: Continue to Phase 6
- Implement explainability (SHAP)
- Add counterfactual generation
- Create explainability UI

---

## Time Estimates

| Task | Time |
|------|------|
| Install Python | 5 minutes |
| Install dependencies | 3-5 minutes |
| Test backend | 2 minutes |
| Train demo model | 2-5 minutes |
| Start servers | 1 minute |
| Frontend setup | 1-2 minutes |
| Browser testing | 2 minutes |

**Total**: ~15-25 minutes

---

## Expected Results

### After Training RoBERTa on Demo Dataset

```json
{
  "model": "roberta_baseline",
  "accuracy": 0.95-1.00,
  "precision": 0.95-1.00,
  "recall": 0.95-1.00,
  "f1": 0.95-1.00
}
```

**Note**: High accuracy is expected on tiny demo dataset (overfitting). Use full dataset for realistic results.

---

## System Requirements

✅ **Confirmed Working**:
- Node.js: v24.11.1 ✅
- npm: 11.6.2 ✅

❌ **Needs Installation**:
- Python: 3.11+ ❌

💾 **Disk Space**: ~2 GB
- PyTorch: ~1 GB
- RoBERTa model: ~500 MB
- Node modules: ~300 MB
- Checkpoints: ~500 MB

🧠 **RAM**: 8 GB minimum (16 GB recommended)

⏱️ **GPU**: Not required (CPU training works, just slower)

---

## Full Testing

For comprehensive validation, see **TESTING_CHECKLIST.md** (30-45 minutes).

---

**Ready to start? Install Python and follow Step 2!**

Repository: https://github.com/aashik2005/EMFRD

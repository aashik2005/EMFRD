# EMFRD - Installation and Running Guide

**Complete step-by-step guide to install and run EMFRD**

---

## ⚠️ Current Status

**Python**: ❌ NOT INSTALLED (Required)  
**Node.js**: ✅ INSTALLED (v24.11.1)  
**System**: Windows

---

## 📋 Prerequisites

### 1. Install Python (REQUIRED - 5 minutes)

**Download & Install:**
1. Go to: https://www.python.org/downloads/
2. Click "Download Python 3.11.x" or "Download Python 3.12.x"
3. Run the installer
4. **⚠️ CRITICAL**: Check "Add Python to PATH"
5. Click "Install Now"
6. Wait for installation to complete
7. **Close and reopen** your terminal/cmd

**Verify Installation:**
```bash
python --version
```
Should show: `Python 3.11.x` or `Python 3.12.x`

---

## 🚀 Quick Start (After Python is Installed)

### Option 1: Automated Setup (Recommended)

```bash
# Run setup (installs all dependencies)
setup.bat

# Run the complete system
run.bat
```

**That's it!** The system will start in ~10 seconds.

Visit: http://localhost:5173

---

### Option 2: Manual Step-by-Step

#### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install Python dependencies (5-10 minutes)
pip install -r requirements.txt

# Install frontend dependencies (2-3 minutes)
cd frontend
npm install
cd ..
```

#### Step 2: Start Backend

```bash
# In terminal 1
venv\Scripts\activate
uvicorn backend.main:app --reload
```

Backend available at: http://localhost:8000

#### Step 3: Start Frontend

```bash
# In terminal 2 (new window)
cd frontend
npm run dev
```

Frontend available at: http://localhost:5173

---

## 📊 Training Models (Optional)

### Quick Test (2-5 minutes)

```bash
# Train on demo dataset
train_demo.bat
```

### Full Training

```bash
# Activate environment
venv\Scripts\activate

# Train each model
python -m backend.training.train_roberta
python -m backend.training.train_contrastive
python -m backend.training.train_hgnn --dataset fraud_amazon
python -m backend.training.train_gan
python -m backend.training.train_fusion
```

---

## 🧪 Testing the System

### 1. Check Backend Health

Open browser: http://localhost:8000/health

Should see:
```json
{
  "status": "healthy",
  "cuda_available": false,
  "device": "cpu"
}
```

### 2. Check API Documentation

Open browser: http://localhost:8000/docs

Should see Swagger API documentation with all endpoints.

### 3. Test Frontend

Open browser: http://localhost:5173

Should see the EMFRD dashboard.

### 4. Make a Prediction

1. Go to "Prediction" page
2. Enter test review: "This product is AMAZING!!!"
3. Select model: "RoBERTa Baseline"
4. Click "Predict"
5. Should see prediction result

---

## 📁 Available Scripts

| Script | Description |
|--------|-------------|
| `setup.bat` | Complete automated setup |
| `run.bat` | Start both backend and frontend |
| `start_backend.bat` | Start backend only |
| `start_frontend.bat` | Start frontend only |
| `train_demo.bat` | Quick training on demo dataset |

---

## 🔧 Troubleshooting

### "Python was not found"

**Solution**: Install Python and make sure "Add Python to PATH" is checked.

After installation, close and reopen terminal.

### "pip is not recognized"

**Solution**: 
```bash
python -m pip install --upgrade pip
```

### "ModuleNotFoundError: No module named 'torch'"

**Solution**: Dependencies not installed
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### "Port 8000 already in use"

**Solution**: Another process is using port 8000
```bash
# Find process
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Backend starts but shows errors

**Solution**: Check if virtual environment is activated
```bash
venv\Scripts\activate
```

### Frontend won't start

**Solution**: 
```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```

---

## 📊 What to Expect

### First Run (No Training)

- Backend starts successfully ✅
- Frontend loads ✅
- API responds ✅
- Predictions work ✅ (but with untrained model warning)

### After Training

- Predictions use trained model ✅
- Higher accuracy ✅
- Confidence scores meaningful ✅
- Experiments page shows results ✅

---

## 🎯 Complete Workflow

### 1. Install Python (5 min)
- Download from python.org
- Install with "Add to PATH"
- Verify: `python --version`

### 2. Run Setup (10-15 min)
```bash
setup.bat
```

### 3. Quick Test (2-5 min)
```bash
train_demo.bat
```

### 4. Start System (1 min)
```bash
run.bat
```

### 5. Test in Browser
- Open: http://localhost:5173
- Go to Prediction page
- Test some reviews
- Check experiments page

---

## 📈 System Architecture

```
User Browser (http://localhost:5173)
         ↓
   React Frontend (Vite Dev Server)
         ↓
   FastAPI Backend (http://localhost:8000)
         ↓
   PyTorch Models (CPU/GPU)
         ↓
   Predictions & Explanations
```

---

## 🔗 Important URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | Main UI |
| **API** | http://localhost:8000 | REST API |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Health** | http://localhost:8000/health | Health check |

---

## 💾 Disk Space Requirements

- Python + Dependencies: ~1.5 GB
- Node.js modules: ~300 MB
- RoBERTa model cache: ~500 MB
- Trained checkpoints: ~500 MB

**Total**: ~2.8 GB

---

## ⚡ Performance

### Demo Dataset (40 samples)
- Training: 2-5 minutes (CPU)
- Prediction: <1 second

### Full Dataset (40K samples)
- Training: 1-8 hours depending on model (CPU)
- Training: 1-2 hours (GPU)
- Prediction: <1 second

---

## 🎓 Next Steps After Setup

1. ✅ **Verify System Works**
   - Run through quick test
   - Make some predictions
   - Check API docs

2. ✅ **Train Models**
   - Start with demo dataset
   - Move to full dataset
   - Train all 5 models

3. ✅ **Explore Features**
   - Try explainability
   - Check experiments
   - Compare models

4. ✅ **Customize**
   - Add your datasets
   - Modify models
   - Extend features

---

## 📞 Help

- **Documentation**: See docs/ folder
- **Issues**: GitHub Issues
- **Quick Guide**: QUICKSTART.md
- **Testing**: TESTING_CHECKLIST.md

---

## ✅ Checklist

Before running:
- [ ] Python 3.11+ installed
- [ ] Python in PATH (verify with `python --version`)
- [ ] Node.js installed (already ✅)
- [ ] Enough disk space (~3 GB)

To start system:
- [ ] Run `setup.bat` (first time only)
- [ ] Run `run.bat`
- [ ] Open http://localhost:5173

---

**Ready to start? Install Python, then run `setup.bat`!**

Repository: https://github.com/aashik2005/EMFRD

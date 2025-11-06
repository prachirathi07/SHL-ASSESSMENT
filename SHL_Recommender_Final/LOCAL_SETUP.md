# Local Development Setup Guide

## Prerequisites
- Python 3.8+ installed
- Node.js 18+ and npm installed
- Virtual environment activated (optional but recommended)

## Step-by-Step Commands

### 1. Navigate to the project directory
```powershell
cd SHL_Recommender_Final
```

### 2. Setup Backend (Python/FastAPI)

#### Activate virtual environment (if using one)
```powershell
# If you have a venv in the parent directory
..\venv\Scripts\Activate.ps1

# Or create a new one
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Install Python dependencies
```powershell
pip install -r requirements.txt
```

#### Run the backend server
```powershell
python run.py
```

The backend will start on **http://localhost:5000**

### 3. Setup Frontend (Next.js)

Open a **new terminal window** (keep the backend running)

#### Navigate to frontend directory
```powershell
cd SHL_Recommender_Final\frontend
```

#### Install Node.js dependencies
```powershell
npm install
```

#### Run the frontend development server
```powershell
npm run dev
```

The frontend will start on **http://localhost:3000**

### 4. Access the Application

Open your browser and go to: **http://localhost:3000**

## Quick Start (All Commands)

**Terminal 1 (Backend):**
```powershell
cd SHL_Recommender_Final
pip install -r requirements.txt
python run.py
```

**Terminal 2 (Frontend):**
```powershell
cd SHL_Recommender_Final\frontend
npm install
npm run dev
```

## Stopping the Servers

- Press `Ctrl+C` in each terminal to stop the servers

## Testing the Backend

You can test if the backend is running by visiting:
- http://localhost:5000/api/health
- http://localhost:5000 (direct FastAPI interface)


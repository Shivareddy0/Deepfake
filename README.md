# Deepfake Shield 🛡️
### High-Performance Deepfake Detection & C2PA Media Provenance Ledger

Deepfake Shield is a production-grade, multi-modal forensics platform designed to verify digital media authenticity and detect synthetic manipulations. The system combines cryptographic C2PA metadata verification, frequency domain anomaly detection, spatial CNN classifiers with adversarial defenses, temporal coherence tracking, and acoustic vocal tract analysis.

---

## 🏗️ Project Architecture
The application is structured as a decoupled system:
- **Backend**: Python FastAPI service running forensic detectors, tiered routing, and a Bayesian Meta-Classifier.
- **Frontend**: Next.js (React) Enterprise Dashboard & Studio for interactive forensic analysis and ledger inspection.

---

## 🚀 Getting Started on Localhost

Follow these step-by-step instructions to set up and run the entire application on your system.

### 📋 Prerequisites
Ensure you have the following installed on your machine:
- **Python** (version `3.8` to `3.11` recommended)
- **Node.js** (version `18.x` or higher) & **npm**

---

### 📥 1. Clone or Extract the Project
Make sure the files are placed in a directory on your machine. Open a terminal (such as Command Prompt, PowerShell, or Bash) in the project root folder.

---

### 🐍 2. Backend Setup (FastAPI Server)

1. **Navigate to the backend folder**:
   ```bash
   cd backend
   ```

2. **Create a Python Virtual Environment**:
   * **Windows**:
     ```bash
     python -m venv venv
     ```
   * **macOS / Linux**:
     ```bash
     python3 -m venv venv
     ```

3. **Activate the Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   * **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Backend Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Start the FastAPI Server**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   * The backend will run on **`http://localhost:8000`**.
   * You can view the automated Swagger documentation at **`http://localhost:8000/docs`**.

> [!NOTE]
> On the first run, the backend will automatically generate the mock blockchain ledger (`backend/c2pa/simulated_ledger.json`) and the model registry (`backend/retraining/model_registry.json`) file databases. No external databases (e.g., PostgreSQL or MongoDB) are needed.

---

### 💻 3. Frontend Setup (Next.js Application)

1. **Open a new terminal window** and navigate to the `frontend/` directory from the project root:
   ```bash
   cd frontend
   ```

2. **Install Node Modules**:
   ```bash
   npm install
   ```

3. **Run the Next.js Development Server**:
   ```bash
   npm run dev
   ```
   * The frontend will start running on **`http://localhost:3000`**.

4. **Open the Web Application**:
   * Open your browser and go to **`http://localhost:3000`** to access the Dashboard.

---

## 🛠️ Features to Try

1. **Enterprise Dashboard**: Review global synthetic media rates, scans count, and C2PA credential verification health.
2. **Deep Analysis Studio (Workbench)**: Upload an image or video to run the tiered detection pipeline and view the GradCAM++ spatial heatmap attention layers.
3. **Audio Forensics Studio**: Test voice tracks or TTS inputs, viewing formant trackers and Mel-spectrogram analyses.
4. **C2PA Provenance Ledger**: Inspect cryptographic proof of authenticity and mock blockchain anchor transactions.

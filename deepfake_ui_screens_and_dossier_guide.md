# Deepfake Shield Screen-by-Screen Walkthrough & Presentation Guide
### A Step-by-Step Slide deck Outline for MNC Office Presentations
*Prepared for Shiva Reddy & team to share with colleagues and build the project PowerPoint (PPT)*

---

## Slide 1: Title & Presentation Intro
*   **Slide Title**: Multi-Signal Deepfake Detection & Media Provenance Ledger
*   **Subtitle**: Production-Grade Verification of Generative AI Media & C2PA Certificates
*   **Key Themes**: Decentralized Trust, Real-Time Forensics, Multi-Model Ensemble, Bayesian Inference.
*   **Presenter Notes**: Briefly introduce the core problem: Deepfakes are evolving faster than individual models can catch them. We have built an integrated platform combining CPU-fast signal math, deep CNN spatial networks, temporal/audio analyzers, and cryptographic provenance checks.

---

## Slide 2: Global Navigation System & Theme Engine
*   **UI Component**: **The App Sidebar Navigation** ([SidebarNav.tsx](file:///d:/shiva%20reddy%20project/Deepfake/frontend/src/components/SidebarNav.tsx))
*   **Screen Layout**: Left-aligned sleek persistent navigation sidebar.
*   **Key Sections**:
    1.  **Brand Logo**: *AetherShield / Deepfake Shield* emblem.
    2.  **Navigation Targets**: 
        *   📊 **Forensic Overview**: Core dashboard.
        *   🔬 **Analyst Workbench**: High-fidelity granular analysis workspace.
        *   ⛓️ **Provenance Viewer**: X.509 certificates and C2PA timeline.
        *   🗃️ **Model Registry & Zoo**: Active retraining logs and accuracy telemetry.
        *   🗂️ **Batch Pipeline Panel**: Asynchronous S3 list ingestion.
    3.  **Utility Controls**: 
        *   ☀️/🌙 **Light-Dark Toggle**: Tailored glassmorphism theme shifter.
        *   🟢 **System Status indicator**: Active WebSocket / FastAPI heartbeat.
*   **Technical Explanation**: The navigation acts as the main shell. It dynamically tracks active states and provides uniform dark/light styling tokens to prevent visual mismatches across pages.

---

## Slide 3: Screen 1 — Forensic Overview (Dashboard)
*   **UI Component**: **Main Dashboard UI** ([page.tsx](file:///d:/shiva%20reddy%20project/Deepfake/frontend/src/app/page.tsx))
*   **Layout**: Top action bar ➔ Grid of 4 Metric cards ➔ Two-column visualization split ➔ Bottom Ledger table.
*   **Section-by-Section Details**:
    1.  **Ingest Media File (Action Button)**: Triggers file picker for images/videos/audio, sending them to the FastAPI `/detect/upload` endpoint.
    2.  **Summary Cards**:
        *   *Total Analyzed*: Counter of all active uploads.
        *   *Deepfake Alerts*: Percentage rate of flagged assets.
        *   *Inference Mode*: Indicator verifying local server connection.
        *   *C2PA Verified*: Percentage of uploads carrying valid cryptographic assertions.
    3.  **Detection Trajectory (SVG Chart)**: Dynamic path graph. The **Cyan line** represents the raw confidence score trends, and the **Rose line** represents binary alert peaks. Contains safety checks to prevent `NaN` coordinates.
    4.  **Signal Breakdown (Attribution Bars)**: Horizontal sliders allocating deepfake causes: Spatial CNN cues, Fourier spectral check, Vocal tract LPC violations, and eye blinks/temporal drifting.
    5.  **Forensic Detection Ledger (Database Table)**: Displays filename, media format, verdict (AI Flagged vs. Authentic), confidence scores, executed tiers, signature checks, and links each item directly into the Analyst Workbench.

---

## Slide 4: Screen 2 — Analyst Workbench (Deep Analysis View)
*   **UI Component**: **Forensic Interactive Studio** ([workbench/page.tsx](file:///d:/shiva%20reddy%20project/Deepfake/frontend/src/app/workbench/page.tsx))
*   **Layout**: Top control bar ➔ Left large Canvas viewport with overlay controls ➔ Right sidebar containing FFT panels, Forensic logs, and EXIF attributes.
*   **Section-by-Section Details**:
    1.  **Canvas Viewport (Interactive Player)**:
        *   *For Images/Videos*: Displays spatial boundaries. Renders green box (`SECURE`) or red box (`AI MANIPULATED`).
        *   *GradCAM++ Overlays*: Heatmap highlighting manipulated face parts (eyes, nose, mouth edges) based on backpropagated CNN gradients.
        *   *3D Face Mesh Check*: Facial landmarks mapped on-canvas showing symmetry checks.
        *   *For Audio*: Draws oscillating raw waveforms and pitch contours (Hz) in real time.
    2.  **Ingestion Controls**:
        *   *File Upload & Cache Select*: Upload local media or quickly load previous analysis files.
        *   *Webcam Ingest (WebRTC)*: Captures live video feed. Binds active canvas frame loops at 30fps with landmark check overlays.
        *   *Microphone Ingest (WebRTC)*: Captures live audio. Binds `AudioContext` and draws real-time RMS amplitude charts.
    3.  **FFT Azimuthal Power Spectrum Viewer**: Draws the 2D Fourier Magnitude Spectrum centered at DC frequency. If a file is AI-generated, **red grids/dots** are plotted, showing GAN upsampling deconvolution checkerboard residuals.
    4.  **Forensic Report Log**: Renders a detailed text block. If the file is fake, it highlights the **Explainable Anomaly Log**:
        *   *"CRITICAL ANOMALY: Manipulation detected in the left eye region (frames 142-189) with frequency anomaly score 4.2σ above natural baseline; facial landmark symmetry violation of 3.7° in jaw angle; lip-audio desynchronization of 120ms starting at timestamp 00:01:23."*
    5.  **Signature / Header Metadata**: Key-value grid of EXIF data (e.g. Editing software signatures, camera sensor profiles).
    6.  **Download Signed Dossier (Button)**: Downloads a text-based forensic dossier (.txt) summarizing all verified assertions.

---

## Slide 5: Screen 3 — C2PA Provenance Viewer
*   **UI Component**: **Cryptographic Authenticity Timeline** ([provenance/page.tsx](file:///d:/shiva%20reddy%20project/Deepfake/frontend/src/app/provenance/page.tsx))
*   **Layout**: Search selector ➔ Left two-column Interactive Timeline ➔ Right Column Provenance Proofs.
*   **Section-by-Section Details**:
    1.  **Scanned Cache Selector**: Dropdown to select scanned files with C2PA metadata.
    2.  **Cryptographic Provenance Chain**: Timeline tracker with icons (e.g., Camera, Cpu).
        *   *Asset Ingestion*: Details manual upload and SHA-256 hashing.
        *   *AetherShield Pipeline Run*: Traces the Bayesian core execution details, confidence, and uncertainty ratios.
    3.  **Provenance Envelope**: Displays the payload hash, blockchain transaction hash, anchored block, and CA issuer details.
    4.  **X.509 Certificate Chain Details**: Expandable section listing signing algorithms (Ed25519), certificate serials, and validity windows.
    5.  **Manifest JSON Dump**: Pretty-printed JSON block showing the exact JUMBF structure, recorder agents, and signed assertions.

---

## Slide 6: Screen 4 — Model Zoo & Telemetry Control
*   **UI Component**: **Model Zoo Manager** ([model-zoo/page.tsx](file:///d:/shiva%20reddy%20project/Deepfake/frontend/src/app/model-zoo/page.tsx))
*   **Layout**: Horizontal Model cards ➔ Right-aligned Training Telemetry log console.
*   **Section-by-Section Details**:
    1.  **Model Cards Grid**: Lists each detector in the backend registry (FFT Analyzer, EfficientNet CNN, Temporal Tracker, Audio ResNet, Header Forensics) along with:
        *   *Version tags* (e.g., `v4.3.0`, `v2.1.0`).
        *   *ROC-AUC*, *Accuracy*, and *False Positive Rate* stats.
        *   *Drift Detection Alert*: Flashes red if model performance drops below the threshold.
    2.  **Retrain Model (Manual override trigger)**: Sends a trigger to the backend API (`/zoo/retrain`) to boot retraining.
    3.  **Retraining Telemetry Console**: A dark CLI console printing active training epochs, mixed-precision FP16 operations, PGD-10 adversarial training processes, and canary suite evaluations in real time.

---

## Slide 7: Screen 5 — Batch Pipeline Panel
*   **UI Component**: **High-Volume Ingestion Panel** ([batch/page.tsx](file:///d:/shiva%20reddy%20project/Deepfake/frontend/src/app/batch/page.tsx))
*   **Layout**: Left Ingestion input form ➔ Right Asynchronous Queue Monitor list.
*   **Section-by-Section Details**:
    1.  **Ingestion Form**:
        *   *Payload URLs Textarea*: Input list of S3 paths or cloud links (one per line).
        *   *Queue Priority dropdown*: Choose Low (T1 background), Medium (T1-T2), or High (immediate T3 audit).
        *   *Webhook Callback Input*: Target endpoint URL for final results.
    2.  **Asynchronous Pipeline Logs**: List of active and completed jobs.
        *   *Job Headers*: Displays Job IDs (e.g., `BATCH-2026-9284`), timestamp, and priority.
        *   *Progress Tracker*: Real-time progress bar (e.g., 8/8 files completed).
        *   *Expandable Files Ledger*: Shows individual filenames, progress statuses, deepfake percentages, and final verdicts.

---

## Slide 8: Technical Deep Dive — The PDF/Signed Dossier
*   **The Dossier Structure**:
    *   **File Parameters**: Filename, payload type, timestamp, routing tiers.
    *   **Ensemble Verdict**: Fused probability, model uncertainty %, C2PA signature, and blockchain anchor status.
    *   **Explainable Log**: Granular details including eye region coordinates (frames 142-189), radial PSD peak σ, jaw symmetry angles, and audio-lip desynchronization timings (ms).
    *   **Header Metadata**: Full EXIF variables (Make, Model, Software, editing traces).
    *   **Cryptographic proof**: SHA-256 bind hash, Hyperledger/Ethereum transaction, CA authorization stamps.
*   **Presenter Tips**: Emphasize this as the core business value. It allows journalists, legal teams, and security analysts to export a verified forensic report suitable for courtrooms or media audits.

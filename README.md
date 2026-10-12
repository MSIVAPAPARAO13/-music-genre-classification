# Music Genre Classification using a Leakage-Safe Hybrid CRNN Model

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-Leakage--Safe%20CRNN-green.svg)](#model-architecture)

## Overview

This repository provides an end-to-end Machine Learning pipeline and interactive Streamlit web application for music genre classification across 10 target genres: **blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock**.

The core system uses a **Hybrid CRNN (Convolutional Recurrent Neural Network)** combining 4 convolutional blocks, self-attention temporal weighting, and bidirectional LSTMs. The model was trained and evaluated under **strict track-level isolation**, completely eliminating historical chunk-overlap data leakage.

* **GitHub Repository**: [https://github.com/MSIVAPAPARAO13/-music-genre-classification](https://github.com/MSIVAPAPARAO13/-music-genre-classification)
* **Live Demo**: *(Deployable via Streamlit Community Cloud — see deployment instructions below)*

---

## Features

* **Premium Audio Intelligence Dashboard**: Modern music-tech UI inspired by Spotify and Apple Music with dark theme, glassmorphic surfaces, emerald accents, and responsive layout.
* **Leakage-Safe Inference Engine**: Production default model (`Leakage_Safe_CRNN_Model.keras`) trained on track-level partitions with zero song overlap.
* **Instant Rollback Support**: Legacy baseline model (`Hybrid_CRNN_Model.keras`) preserved and selectable in real time via the app's sidebar.
* **Explainable AI via Grad-CAM**: Visualizes convolutional saliency maps over Mel-spectrogram chunks to highlight time-frequency activations influencing the model's decision.
* **Multi-Chunk Trajectory & Dissenting Chunks**: Inspects chunk-by-chunk consensus, time windows, and highlights dissenting chunks during musical transitions.
* **Full Audio Visualizations**: Interactive amplitude waveform over time, full log Mel-spectrogram (dB), and individual 4-second chunk spectrogram inspector.
* **Dual Format Audio Support**: Accepts both MP3 and WAV file uploads with automatic native decoding.
* **Acoustically Aligned Preprocessing**: Resamples all incoming audio to native 22,050 Hz, slices into rolling 4-second chunks with 2-second overlap, and computes raw power Mel spectrograms resized to `(150, 150, 1)`.
* **Dedicated Benchmark Performance Dashboard**: Visualizes GTZAN held-out test benchmarks, per-genre precision/recall/F1, and a 10×10 confusion matrix heatmap.
* **Robust Validation**: Gracefully catches and explains unreadable files, corrupt streams, empty audio, and clips under 4.0 seconds.

---

## Model Architecture & Preprocessing

```
Audio Input (MP3 / WAV)
       │
       ▼
Resample to 22,050 Hz ──► Validate Duration (>= 4.0s)
       │
       ▼
Extract Rolling 4s Chunks (2s overlap, non-stretched)
       │
       ▼
Compute Mel Spectrogram (128 mel bins, n_fft=2048, hop_length=512)
       │
       ▼
Resize Spectrogram to (150, 150, 1) float32
       │
       ▼
┌────────────────────────────────────────────────────────┐
│             Hybrid CRNN Architecture                   │
├────────────────────────────────────────────────────────┤
│ Conv Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool(2x2) │
│ Conv Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool(2x2) │
│ Conv Block 3: Conv2D(128)-> Conv2D(128)-> MaxPool(2x2)│
│ Conv Block 4: Conv2D(256)-> Conv2D(256)-> MaxPool(2x2)│
│ Reshape to Temporal Sequence (49 time steps, 256 dim)  │
│ Dense Attention Weights Layer -> Elementwise Multiply   │
│ Bidirectional LSTM (256 units, return sequences)       │
│ Bidirectional LSTM (128 units)                         │
│ Fully Connected Dense(256) -> BatchNorm -> Dropout     │
│ Dense(10, Softmax) Output                              │
└────────────────────────────────────────────────────────┘
       │
       ▼
Chunk Probabilities Average ──► Top-3 Genre Ranking
```

---

## Dataset & Leakage-Safe Split Methodology

The model is trained on the benchmark **GTZAN Genre Collection**:
* 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.
* 1,000 original 30-second clips at 22,050 Hz mono.
* Track `jazz.00054.wav` is a known corrupt file and is excluded, leaving **999 valid tracks**.

### Why Track-Level Partitioning Matters
In historical literature and earlier iterations, songs were sliced into 4-second overlapping chunks *before* splitting. Because adjacent chunks share 50% identical samples and acoustic features, chunk-level random splitting caused severe data leakage (adjacent chunks of the test song were present in the training set), yielding an artificially inflated ~94–96% accuracy.

In this pipeline:
1. **Track-Level Split**: Songs are partitioned into Train (70%), Validation (15%), and Test (15%) **before** chunking.
2. **Zero Track Overlap**: No song appears in more than one partition ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
3. **Partition Manifest**: Codified reproducibly in `gtzan_leakage_safe_split.json`.
4. **Checkpoint Selection**: Model weights selected solely based on validation accuracy (Epoch 10: 72.45%).

---

## Verified Held-Out Test Results

Evaluated strictly **once** on the 150 completely unseen held-out test tracks (2,100 chunks):

| Evaluation Metric | Value | Details |
| :--- | :--- | :--- |
| **Track-Level Accuracy** | **81.33%** | 122 / 150 unseen songs correctly classified |
| **Macro Precision** | **82.34%** | Average precision across all 10 genres |
| **Macro Recall** | **81.33%** | Average recall across all 10 genres |
| **Macro F1-Score** | **80.93%** | Balanced harmonic mean |
| **Chunk-Level Accuracy** | **74.71%** | 1,569 / 2,100 chunks correctly classified |

### Per-Genre Performance on Held-Out Test Songs (15 tracks / genre)

| Genre | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Metal** | 0.88 | **1.00** | **0.94** | 15 |
| **Classical** | 0.93 | **0.93** | **0.93** | 15 |
| **Reggae** | **1.00** | 0.87 | **0.93** | 15 |
| **Hiphop** | **1.00** | 0.80 | **0.89** | 15 |
| **Blues** | 0.87 | 0.87 | **0.87** | 15 |
| **Disco** | 0.75 | **1.00** | **0.86** | 15 |
| **Jazz** | 0.67 | **0.93** | **0.78** | 15 |
| **Country** | 0.69 | 0.60 | **0.64** | 15 |
| **Pop** | 0.80 | 0.53 | **0.64** | 15 |
| **Rock** | 0.64 | 0.60 | **0.62** | 15 |
| **Macro Avg** | **0.82** | **0.81** | **0.81** | **150** |

*Full metrics and confusion matrix are stored in `leakage_safe_evaluation_report.json`.*

---

## Generalization Boundaries & Limitations

* **Benchmark Specificity**: Trained on GTZAN, which consists of audio excerpts recorded in the late 1990s and early 2000s.
* **Acoustic Domain Shifts**: Modern commercial tracks with heavy dynamic range compression, unique mixing, or emerging electronic subgenres may differ from GTZAN acoustic patterns.
* **Softmax Scores**: Displayed confidence percentages reflect softmax activations averaged across chunks, representing relative acoustic affinity rather than calibrated real-world probabilities.
* **Rollback Option**: If comparison is required, users can switch to the legacy `Hybrid_CRNN_Model.keras` checkpoint in the application sidebar.

---

## Local Installation & Execution

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/MSIVAPAPARAO13/-music-genre-classification.git
cd -music-genre-classification

# Create virtual environment (Python 3.10 - 3.12 recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run Music_Genre_App.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Automated Tests
```bash
pytest test_inference.py -v
```

---

## Public Deployment Instructions

### Option A: Streamlit Community Cloud (Recommended)
1. Push repository changes to GitHub (`main` branch).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Configure:
   * **Repository**: `MSIVAPAPARAO13/-music-genre-classification`
   * **Branch**: `main`
   * **Main file path**: `Music_Genre_App.py`
4. The deployment automatically utilizes:
   * `requirements.txt` for Python packages (`streamlit`, `tensorflow`, `librosa`, etc.).
   * `packages.txt` for Linux system libraries (`libsndfile1`, `ffmpeg`).
5. Once live, test audio file upload and prediction.

### Option B: Docker Container
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "Music_Genre_App.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t music-genre-app .
docker run -p 8501:8501 music-genre-app
```

---

## Project Structure

```
├── .streamlit/config.toml             # Streamlit dark theme & production server config
├── Music_Genre_App.py                 # Streamlit web application & inference pipeline
├── Leakage_Safe_CRNN_Model.keras      # Default production model (Phase 3B leakage-safe)
├── Hybrid_CRNN_Model.keras            # Legacy production checkpoint (rollback option)
├── leakage_safe_evaluation_report.json# Held-out test evaluation report & metrics
├── leakage_safe_training_hist.json    # Epoch-by-epoch training and validation loss/accuracy
├── leakage_safe_training.py           # Reproducible training & evaluation pipeline
├── gtzan_leakage_safe_split.json      # Frozen 70/15/15 track-level partition manifest
├── test_inference.py                  # Pytest test suite (22 automated unit tests)
├── requirements.txt                   # Pinned Python package dependencies
├── packages.txt                       # Linux system packages (libsndfile1, ffmpeg)
├── Images/home.jpg                    # Application banner image
├── Test_music/                        # Real audio test samples for verification
└── README.md                          # Documentation
```

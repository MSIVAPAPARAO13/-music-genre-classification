# Music Genre Classification using a Hybrid CRNN Model

<!-- [![Live App](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://music-genre-classification-crnn.streamlit.app/) -->

## Overview

This project focuses on classifying music into 10 different target genres using advanced Deep Learning techniques. The current iteration utilizes a high-performance **Hybrid CRNN (Convolutional Recurrent Neural Network)** embedded with an attention mechanism to analyze audio signals. By breaking audio into rolling 4-second chunks and converting them into high-resolution Mel spectrograms, the model accurately identifies complex spatio-temporal audio patterns. 

<!-- You can try the live application here: **[Music Genre Classifier App](https://music-genre-classification-crnn.streamlit.app/)** -->

## Features

* **Hybrid CRNN Engine**: Uses a 4-block CNN mated to a Bi-directional LSTM and an Attention layer for state-of-the-art accuracy.
* **Robust Prediction Logic**: Processes 30-second audio files sequentially by overlapping 4-second chunks and calculating a mean probability score across all segments for high-confidence predictions.
* **Interactive Dashboard**: A simple and functional user interface built with Streamlit.
* **Spectrogram Analysis**: Translates waveforms into visual features (150x150x1 Mel Spectrograms) enabling powerful image-based learning.

## Dataset

* **GTZAN Dataset** ([Download on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification))
  * 10 genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock)
  * 100 audio files per genre
  * Each audio clip is 30 seconds long

## Tech Stack

* Python
* Librosa (audio processing)
* TensorFlow / Keras (Neural Network design & inference)
* Streamlit (Web UI)
* NumPy, Pandas, Matplotlib

## Methodology

1. **Audio Loading**: Load the `.mp3` or `.wav` waveform data.
2. **Chunking**: Slice the audio into 4-second rolling chunks with a 2-second overlap to ensure no audio context is lost.
3. **Spectrogram Generation**: Convert the audio chunks into Mel Spectrogram images and normalize them to a target shape of `(150, 150, 1)`.
4. **Prediction Algorithm**: The CRNN model generates raw probabilities for each chunk. We calculate the mathematical average (mean confidence) across all chunks to determine the overall certainty.
5. **Final Inference**: The genre class with the highest mean probability is selected as the final predicted genre.

## Model Architecture

The custom `Hybrid_Custom_CRNN` model pipeline includes:
* **Feature Extraction**: 4 Sequential Convolutional/Max-Pooling Blocks (`Conv2D` -> `BatchNorm` -> `MaxPooling2D`).
* **Temporal Sequence Mapping**: Bi-directional LSTM (`Bidirectional(LSTM)`) to learn the long-term context/flow of the song.
* **Contextual Focus**: Custom Context Attention Layer to prioritize critical musical events.
* **Classification**: Fully Connected Dense Layers ending smoothly in a 10-unit Softmax activation.

## Results

* Reached an impressive **Validation Accuracy of 96.39%** utilizing the Hybrid CRNN architecture.
* Extremely robust real-world performance by averaging chunk predictions to prevent random temporal noise from skewing the final result.

## How to Run Locally

Clone the repository and install dependencies:
```bash
git clone https://github.com/mandalrs/music-genre-classification.git
cd music-genre-classification
pip install -r requirements.txt
```

Launch the web app:
```bash
streamlit run Music_Genre_App.py
```

*Note: You can inspect the training, testing, and AI pipeline by opening `CRNN_TRAIN.ipynb` and `CRNN_TEST.ipynb` in a Jupyter environment.*

## Future Improvements

* Real-time BPM and tempo detection mapping
* Premium UI redesign
* Mobile deployment using TensorFlow Lite

---

## Contribution

Feel free to fork this repository and contribute!

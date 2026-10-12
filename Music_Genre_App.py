import os
import io
import json
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.image import resize

# ---------------------------------------------------------
# Global Configuration & Model Paths
# ---------------------------------------------------------
TARGET_SR = 22050
CHUNK_DURATION = 4      # seconds
OVERLAP_DURATION = 2    # seconds
TARGET_SHAPE = (150, 150)
GENRE_LABELS = [
    'blues', 'classical', 'country', 'disco',
    'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "Leakage_Safe_CRNN_Model.keras")
ROLLBACK_MODEL_PATH = os.path.join(BASE_DIR, "Hybrid_CRNN_Model.keras")

AVAILABLE_MODELS = {
    "Leakage-Safe CRNN (Production - Recommended)": PRIMARY_MODEL_PATH,
    "Legacy Hybrid CRNN (Rollback Option)": ROLLBACK_MODEL_PATH,
}

SAMPLE_TRACKS = {
    "🎸 Rock Track — Driving Guitars & Riff (music_rock.mp3)": os.path.join(BASE_DIR, "Test_music", "music_rock.mp3"),
    "🎻 Classical Track — Orchestral Symphony (music_classical.mp3)": os.path.join(BASE_DIR, "Test_music", "music_classical.mp3"),
    "🪩 Disco Track — Upbeat Dance Groove (music_disco.mp3)": os.path.join(BASE_DIR, "Test_music", "music_disco.mp3"),
    "🎤 Hip-Hop Track — 808 Beats & Bass (music_hiphop2.mp3)": os.path.join(BASE_DIR, "Test_music", "music_hiphop2.mp3"),
    "🎷 Jazz Track — Brass Ensemble & Swing (music_jazz2.mp3)": os.path.join(BASE_DIR, "Test_music", "music_jazz2.mp3"),
    "🤠 Country Track — Melodic Acoustic Guitar (test_music2.mp3)": os.path.join(BASE_DIR, "Test_music", "test_music2.mp3"),
}


# ---------------------------------------------------------
# Model Loading & Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model(model_path: str = PRIMARY_MODEL_PATH):
    """
    Load Keras CRNN model.
    Defaults to 'Leakage_Safe_CRNN_Model.keras' (trained with zero track-level leakage).
    Automatically falls back to 'Hybrid_CRNN_Model.keras' if the primary model is absent.
    """
    target_path = model_path
    if not os.path.exists(target_path):
        if os.path.exists(ROLLBACK_MODEL_PATH):
            target_path = ROLLBACK_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Model file not found at '{model_path}' and fallback '{ROLLBACK_MODEL_PATH}' is missing."
            )
    return tf.keras.models.load_model(target_path, compile=False)


@st.cache_data
def load_evaluation_report():
    """Load the verified GTZAN held-out test evaluation report."""
    report_path = os.path.join(BASE_DIR, "leakage_safe_evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------
# Audio Preprocessing & Feature Extraction
# ---------------------------------------------------------
def get_audio_metadata(file_path_or_bytes):
    """
    Extract duration, sample rate, channels, and format from audio file.
    Does not modify audio data.
    """
    if hasattr(file_path_or_bytes, "seek"):
        file_path_or_bytes.seek(0)

    try:
        info = sf.info(file_path_or_bytes)
        orig_sr = info.samplerate
        channels = info.channels
        duration = info.duration
        format_str = info.format
    except Exception:
        if hasattr(file_path_or_bytes, "seek"):
            file_path_or_bytes.seek(0)
        y, sr = librosa.load(file_path_or_bytes, sr=None)
        orig_sr = sr
        channels = 1 if y.ndim == 1 else y.shape[0]
        duration = len(y) / sr if y.ndim == 1 else y.shape[-1] / sr
        format_str = "AUDIO"

    return {
        "duration_sec": float(duration),
        "original_sr": int(orig_sr),
        "channels": int(channels),
        "format": str(format_str)
    }


def load_audio_waveform(file_path_or_bytes, target_sr=TARGET_SR):
    """
    Load raw audio waveform resampled to target_sr for visualization.
    """
    if hasattr(file_path_or_bytes, "seek"):
        file_path_or_bytes.seek(0)
    audio_data, sample_rate = librosa.load(file_path_or_bytes, sr=target_sr)
    return audio_data, sample_rate


def load_and_preprocess_file(file_path, target_shape=TARGET_SHAPE, max_duration_sec=120):
    """
    Load an audio file, resample to TARGET_SR (22,050 Hz) to match training data,
    slice into 4-second chunks with 2-second overlap, and extract 150x150 Mel spectrograms.
    Incomplete chunks (< 4.0s), empty audio, and silent audio are rejected.
    Audio longer than max_duration_sec is bounded to prevent unbounded memory growth.
    """
    if hasattr(file_path, "seek"):
        file_path.seek(0)

    try:
        audio_data, sample_rate = librosa.load(file_path, sr=TARGET_SR)
    except Exception as e:
        raise ValueError(f"Failed to read audio file. Please ensure the file is a valid MP3 or WAV audio track: {e}")

    if audio_data is None or len(audio_data) == 0:
        raise ValueError("Audio file is empty or contains no decodable audio stream.")

    # Silence check: reject tracks with zero or imperceptible acoustic energy
    peak_amp = float(np.max(np.abs(audio_data)))
    rms_amp = float(np.sqrt(np.mean(audio_data ** 2)))
    if peak_amp < 1e-4 or rms_amp < 1e-5:
        raise ValueError(
            f"Audio file is silent or contains imperceptible amplitude (Peak: {peak_amp:.1e}, RMS: {rms_amp:.1e}). "
            "Audible musical content is required for genre classification."
        )

    # Bound duration to max_duration_sec to keep memory usage bounded on long files
    if max_duration_sec is not None and len(audio_data) > int(max_duration_sec * sample_rate):
        audio_data = audio_data[:int(max_duration_sec * sample_rate)]

    chunk_samples = CHUNK_DURATION * sample_rate
    overlap_samples = OVERLAP_DURATION * sample_rate
    step_samples = chunk_samples - overlap_samples

    if len(audio_data) < chunk_samples:
        duration_sec = len(audio_data) / sample_rate
        raise ValueError(
            f"Audio duration ({duration_sec:.2f}s) is shorter than the required "
            f"{CHUNK_DURATION}.0 seconds for chunk analysis."
        )

    data = []
    start = 0
    # Extract only complete 4-second chunks to match training without temporal stretching
    while start + chunk_samples <= len(audio_data):
        chunk = audio_data[start : start + chunk_samples]
        if len(chunk) == chunk_samples:
            # Matches CRNN_TRAIN.ipynb: raw power Mel spectrogram without dB scaling or normalization
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_resized)
        start += step_samples

    if len(data) == 0:
        raise ValueError("No complete 4-second chunks could be extracted from this audio file.")

    chunks_array = np.array(data, dtype=np.float32)

    # Sanitize any non-finite values (NaNs, Infs)
    if np.isnan(chunks_array).any() or np.isinf(chunks_array).any():
        chunks_array = np.nan_to_num(chunks_array, nan=0.0, posinf=1.0, neginf=0.0)

    return chunks_array


# ---------------------------------------------------------
# Prediction & Chunk Aggregation Logic
# ---------------------------------------------------------
def model_prediction(X_test, model=None, return_chunk_details=False):
    """
    Run CRNN inference on preprocessed spectrogram chunks, average prediction scores
    across chunks, and return the predicted genre index, top 3 genres with scores,
    and all aggregated scores.

    If return_chunk_details=True, also returns a list of dictionaries with per-chunk
    predictions, top genre, and consensus status.
    """
    if model is None:
        model = load_trained_model()

    probabilities = model.predict(X_test, verbose=0)
    mean_probabilities = np.mean(probabilities, axis=0)

    ranked_indices = np.argsort(mean_probabilities)[::-1]
    top_predictions = [
        {
            "genre": GENRE_LABELS[idx],
            "score": float(mean_probabilities[idx]),
            "rank": rank + 1,
        }
        for rank, idx in enumerate(ranked_indices[:3])
    ]
    predicted_idx = int(ranked_indices[0])

    if not return_chunk_details:
        return predicted_idx, top_predictions, mean_probabilities

    # Per-chunk breakdown
    chunk_details = []
    for i in range(len(probabilities)):
        chunk_probs = probabilities[i]
        chunk_pred_idx = int(np.argmax(chunk_probs))
        chunk_genre = GENRE_LABELS[chunk_pred_idx]
        chunk_conf = float(chunk_probs[chunk_pred_idx])
        start_t = i * (CHUNK_DURATION - OVERLAP_DURATION)
        end_t = start_t + CHUNK_DURATION
        chunk_details.append({
            "chunk_idx": i + 1,
            "start_time": start_t,
            "end_time": end_t,
            "time_window": f"{start_t:.1f}s – {end_t:.1f}s",
            "predicted_genre": chunk_genre,
            "confidence": chunk_conf,
            "probabilities": chunk_probs,
            "agrees_with_overall": (chunk_pred_idx == predicted_idx)
        })

    return predicted_idx, top_predictions, mean_probabilities, chunk_details


# ---------------------------------------------------------
# Explainability: Grad-CAM on Final Conv Layer
# ---------------------------------------------------------
def compute_gradcam(model, chunk_spec, class_idx, layer_name="conv_b4_2"):
    """
    Compute Grad-CAM heatmap for a single (1, 150, 150, 1) spectrogram chunk.
    Targets layer_name (default: 'conv_b4_2' in Leakage_Safe_CRNN and Hybrid_CRNN).
    Returns normalized (150, 150) 2D float array in range [0, 1].
    Handles missing layers, non-gradient models, and NaNs gracefully.
    """
    try:
        # Validate class_idx bounds
        if not (0 <= class_idx < len(GENRE_LABELS)):
            return np.zeros(TARGET_SHAPE, dtype=np.float32)

        # Ensure chunk_spec is shape (1, 150, 150, 1)
        if chunk_spec.ndim == 2:
            chunk_spec = np.expand_dims(chunk_spec, axis=-1)
        if chunk_spec.ndim == 3:
            chunk_spec = np.expand_dims(chunk_spec, axis=0)
        chunk_tensor = tf.convert_to_tensor(chunk_spec, dtype=tf.float32)

        # Locate target layer
        target_layer = None
        for layer in model.layers:
            if layer.name == layer_name:
                target_layer = layer
                break

        # Dynamic fallback to last Conv2D layer if named layer not found
        if target_layer is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer = layer
                    break

        if target_layer is None:
            return np.zeros(TARGET_SHAPE, dtype=np.float32)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(chunk_tensor)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return np.zeros(TARGET_SHAPE, dtype=np.float32)

        # Global average pooling of gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight activation channels by pooled gradients
        cam = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        cam = tf.maximum(cam, 0.0)  # ReLU: retain only features that positively drive class
        max_val = tf.math.reduce_max(cam)
        if max_val > 0:
            cam = cam / max_val

        # Resize to original (150, 150) input resolution
        cam_resized = tf.image.resize(cam[..., tf.newaxis], TARGET_SHAPE)
        cam_out = np.nan_to_num(cam_resized.numpy().squeeze(), nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(cam_out, 0.0, 1.0)
    except Exception:
        return np.zeros(TARGET_SHAPE, dtype=np.float32)


def generate_gradcam_interpretation(cam_map, target_genre, chunk_idx=1, time_window="0.0s - 4.0s"):
    """
    Analyze energy distribution in Grad-CAM saliency map and return structured interpretation.
    Divides the 150 mel frequency bins into Low (0-50, ~0-1.5kHz), Mid (50-100, ~1.5-4.5kHz),
    and High (100-150, ~4.5-11kHz) bands.
    """
    if cam_map is None or np.all(cam_map == 0):
        return {
            "summary": "No positive gradient activations detected for this class/chunk.",
            "dominant_band": "None",
            "low_pct": 0.0, "mid_pct": 0.0, "high_pct": 0.0,
            "temporal_focus": "Uniform",
            "genre_note": "The convolutional feature filters did not exhibit significant positive activation for this genre."
        }

    total_energy = float(np.sum(cam_map))
    if total_energy <= 1e-6:
        total_energy = 1e-6

    # Frequency bin distribution (150 bins: 0 is lowest freq, 150 is highest freq ~11kHz)
    low_energy = float(np.sum(cam_map[:50, :])) / total_energy * 100.0
    mid_energy = float(np.sum(cam_map[50:100, :])) / total_energy * 100.0
    high_energy = float(np.sum(cam_map[100:, :])) / total_energy * 100.0

    # Temporal concentration (first half vs second half of the 4.0s window)
    t_first = float(np.sum(cam_map[:, :75])) / total_energy * 100.0
    t_second = float(np.sum(cam_map[:, 75:])) / total_energy * 100.0

    if abs(t_first - t_second) < 15.0:
        temporal_focus = "Evenly sustained throughout the 4.0-second window"
    elif t_first > t_second:
        temporal_focus = f"Front-loaded ({t_first:.0f}% concentrated in initial 2.0s)"
    else:
        temporal_focus = f"Back-loaded ({t_second:.0f}% concentrated in final 2.0s)"

    bands = [
        ("Low Frequencies (0–1.5 kHz: Bass & Sub-rhythm)", low_energy),
        ("Mid Frequencies (1.5–4.5 kHz: Melodic & Vocals)", mid_energy),
        ("High Frequencies (4.5–11 kHz: Transients & Cymbals)", high_energy)
    ]
    dominant_band, dominant_pct = max(bands, key=lambda x: x[1])
    genre_cap = target_genre.capitalize()

    if "Low" in dominant_band:
        genre_note = f"The model heavily relies on low-end rhythm, basslines, and kick drum energy to identify **{genre_cap}**."
    elif "Mid" in dominant_band:
        genre_note = f"The model is primarily focused on mid-range melodic structures, vocal resonances, or instrument leads characteristic of **{genre_cap}**."
    else:
        genre_note = f"The model is responding strongly to high-frequency cymbal crashes, breathiness, and percussive transients typical of **{genre_cap}**."

    return {
        "summary": f"Strongest sensitivity in **{dominant_band.split(' (')[0]}** ({dominant_pct:.1f}% of total activation).",
        "dominant_band": dominant_band,
        "low_pct": low_energy,
        "mid_pct": mid_energy,
        "high_pct": high_energy,
        "temporal_focus": temporal_focus,
        "genre_note": genre_note
    }


def render_workflow_stepper(stage="awaiting"):
    """
    Renders visual 9-step audio intelligence workflow stepper:
    Upload Audio -> Validate Audio -> Preprocess -> Segment into Chunks ->
    Generate Mel-Spectrograms -> Run CRNN Inference -> Aggregate Scores ->
    Display Genre Results -> Explore Explanations
    """
    steps = [
        ("① Upload Audio", "done" if stage in ["uploaded", "analyzed"] else "active"),
        ("② Validate Audio", "done" if stage in ["uploaded", "analyzed"] else ("active" if stage == "uploaded" else "pending")),
        ("③ Preprocess (22.05 kHz)", "done" if stage == "analyzed" else "pending"),
        ("④ Segment (4s / 2s Stride)", "done" if stage == "analyzed" else "pending"),
        ("⑤ Mel-Spectrograms", "done" if stage == "analyzed" else "pending"),
        ("⑥ CRNN Inference", "done" if stage == "analyzed" else "pending"),
        ("⑦ Aggregate Scores", "done" if stage == "analyzed" else "pending"),
        ("⑧ Display Results", "done" if stage == "analyzed" else "pending"),
        ("⑨ Explore Explanations", "done" if stage == "analyzed" else "pending")
    ]
    html = ['<div class="workflow-strip">']
    for i, (label, st_class) in enumerate(steps):
        icon = "✅ " if st_class == "done" else ("👉 " if st_class == "active" else "⏳ ")
        html.append(f'<span class="wf-step {st_class}">{icon}{label}</span>')
        if i < len(steps) - 1:
            html.append('<span class="wf-arrow">→</span>')
    html.append('</div>')
    return "".join(html)



# ---------------------------------------------------------
# Matplotlib Visualization Helpers (Clean & Memory-Safe)
# ---------------------------------------------------------
def plot_waveform(audio_data, sr):
    """Plot amplitude waveform over time."""
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    times = np.linspace(0, len(audio_data) / sr, len(audio_data))
    ax.plot(times, audio_data, color='#1DB954', lw=0.7, alpha=0.9)
    ax.set_title("Audio Waveform (Amplitude vs. Time)", color='white', fontsize=12, pad=10)
    ax.set_xlabel("Time (seconds)", color='#b0b0b0', fontsize=10)
    ax.set_ylabel("Amplitude", color='#b0b0b0', fontsize=10)
    ax.tick_params(colors='#b0b0b0')
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(audio_data, sr):
    """Plot full log Mel-spectrogram in decibels."""
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    melspec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    ref_val = float(np.max(melspec)) if np.max(melspec) > 0 else 1.0
    log_melspec = librosa.power_to_db(melspec, ref=ref_val)
    duration = len(audio_data) / sr if sr > 0 else 0

    im = ax.imshow(
        log_melspec, aspect='auto', origin='lower',
        cmap='magma', extent=[0, duration, 0, sr // 2]
    )
    cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.ax.tick_params(colors='#b0b0b0')
    if hasattr(cbar, 'outline') and cbar.outline is not None:
        try:
            cbar.outline.set_edgecolor('#333333')
        except Exception:
            pass

    ax.set_title("Log Mel-Spectrogram (Time vs. Frequency Energy)", color='white', fontsize=12, pad=10)
    ax.set_xlabel("Time (seconds)", color='#b0b0b0', fontsize=10)
    ax.set_ylabel("Frequency (Hz)", color='#b0b0b0', fontsize=10)
    ax.tick_params(colors='#b0b0b0')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    fig.tight_layout()
    return fig


def plot_genre_scores_chart(mean_probabilities, predicted_idx):
    """Horizontal bar chart showing model confidence scores across all 10 genres."""
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Ensure 1D numpy array and sanitize NaNs
    mean_probabilities = np.nan_to_num(np.asarray(mean_probabilities).flatten(), nan=0.0)

    # Sort descending
    sorted_order = np.argsort(mean_probabilities)
    sorted_labels = [GENRE_LABELS[i].capitalize() for i in sorted_order]
    sorted_scores = mean_probabilities[sorted_order] * 100

    colors = [
        '#1DB954' if idx == predicted_idx else '#3498db'
        for idx in sorted_order
    ]

    bars = ax.barh(np.arange(len(sorted_labels)), sorted_scores, color=colors, edgecolor='none', height=0.65)
    ax.set_yticks(np.arange(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, color='white', fontsize=10)
    ax.set_xlabel("Aggregated Model Score (%)", color='#b0b0b0', fontsize=10)
    ax.set_title("Aggregated Genre Prediction Scores (All 10 Classes)", color='white', fontsize=12, pad=12)
    ax.tick_params(colors='#b0b0b0')
    ax.grid(axis='x', color='#333333', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    # Value annotations on bar tips
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va='center', ha='left', color='#e0e0e0', fontsize=9
        )
    ax.set_xlim(0, max(sorted_scores.max() + 15, 100))
    fig.tight_layout()
    return fig


def plot_gradcam_overlay(chunk_spec, cam_heatmap, genre_name, chunk_num=1, time_window="0.0s - 4.0s"):
    """Overlay Grad-CAM heatmap on single chunk spectrogram with physical audio axes."""
    chunk_spec = np.squeeze(chunk_spec)
    cam_heatmap = np.nan_to_num(cam_heatmap, nan=0.0, posinf=1.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Display spectrogram base with physical extent [0 to 4.0s, 0 to 11,025 Hz]
    extent = [0.0, 4.0, 0, TARGET_SR // 2]
    ax.imshow(chunk_spec, aspect='auto', origin='lower', cmap='gray', extent=extent)
    # Overlay heatmap with alpha transparency
    im = ax.imshow(cam_heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.55, extent=extent)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#b0b0b0')
    if hasattr(cbar, 'outline') and cbar.outline is not None:
        try:
            cbar.outline.set_edgecolor('#333333')
        except Exception:
            pass
    cbar.set_label("Saliency Weight [0.0 – 1.0]", color='#b0b0b0', fontsize=9)

    ax.set_title(
        f"Grad-CAM Saliency for '{genre_name.capitalize()}' — Chunk #{chunk_num} ({time_window})",
        color='white', fontsize=11, pad=10
    )
    ax.set_xlabel("Time within Window (0.0s – 4.0s)", color='#b0b0b0', fontsize=9)
    ax.set_ylabel("Audio Frequency (0 – 11,025 Hz)", color='#b0b0b0', fontsize=9)
    ax.tick_params(colors='#b0b0b0')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(report_data):
    """Plot 10x10 confusion matrix heatmap from evaluation report."""
    cm = np.array(report_data["confusion_matrix"])
    labels = [l.capitalize() for l in report_data["genre_labels"]]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar=True, ax=ax,
        linewidths=0.5, linecolor='#222222'
    )
    ax.set_title("GTZAN Held-out Test Set Confusion Matrix (150 Tracks)", color='white', fontsize=13, pad=12)
    ax.set_xlabel("Predicted Genre", color='#b0b0b0', fontsize=11)
    ax.set_ylabel("True Genre", color='#b0b0b0', fontsize=11)
    ax.tick_params(colors='#b0b0b0')
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Premium CSS Design System
# ---------------------------------------------------------
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Tokens ───────────────────────────────────────── */
:root {
  --bg-base:        #080c10;
  --bg-surface:     #0d1117;
  --bg-raised:      #111820;
  --bg-card:        rgba(17, 24, 32, 0.85);
  --border-subtle:  rgba(255,255,255,0.06);
  --border-glow:    rgba(16, 185, 129, 0.35);
  --accent:         #10b981;
  --accent-bright:  #34d399;
  --accent-dim:     rgba(16, 185, 129, 0.12);
  --accent-glow:    0 0 20px rgba(16,185,129,0.25);
  --blue:           #3b82f6;
  --blue-dim:       rgba(59,130,246,0.12);
  --text-primary:   #f0f4f8;
  --text-secondary: #94a3b8;
  --text-muted:     #475569;
  --radius-sm:      8px;
  --radius-md:      14px;
  --radius-lg:      20px;
  --radius-xl:      28px;
}

/* ── Base Reset ────────────────────────────────────────── */
.main .block-container {
  padding: 1.5rem 2.5rem 3rem !important;
  max-width: 1280px;
}
html, body, [data-testid="stApp"] {
  background: var(--bg-base) !important;
  font-family: 'Inter', sans-serif;
  color: var(--text-primary);
}

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080c10 0%, #0d1520 100%) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] h1 { display: none; }

.sidebar-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 0 0 1.5rem 0;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 1.5rem;
}
.sidebar-brand-icon {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, var(--accent), #0ea5e9);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  box-shadow: var(--accent-glow);
}
.sidebar-brand-text { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.95rem; line-height: 1.2; }
.sidebar-brand-sub  { font-size: 0.65rem; color: var(--text-muted); font-weight: 400; }

/* ── Selectbox ─────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  transition: border-color .2s;
}
[data-testid="stSelectbox"] > div > div:hover {
  border-color: var(--accent) !important;
}

/* ── Metric Cards ──────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.1rem 1.3rem !important;
  backdrop-filter: blur(12px);
  transition: transform .2s, border-color .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: var(--border-glow) !important;
  box-shadow: var(--accent-glow);
}
[data-testid="stMetricLabel"]  { color: var(--text-secondary) !important; font-size: 0.75rem !important; font-weight: 500; letter-spacing: .05em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: var(--text-primary) !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 1.7rem !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { color: var(--accent-bright) !important; font-size: 0.78rem !important; font-weight: 600; }

/* ── Tabs ──────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg-raised);
  border-radius: var(--radius-md) !important;
  padding: 4px !important;
  border: 1px solid var(--border-subtle);
  gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: 10px !important;
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  padding: 6px 14px !important;
  transition: all .2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent), #0ea5e9) !important;
  color: #fff !important;
  box-shadow: 0 2px 12px rgba(16,185,129,.35);
}

/* ── Buttons ───────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #10b981, #0ea5e9) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  padding: 0.65rem 2rem !important;
  box-shadow: 0 4px 20px rgba(16,185,129,.4);
  transition: transform .15s, box-shadow .15s;
  letter-spacing: .03em;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(16,185,129,.55) !important;
}

/* ── File Uploader ─────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg-raised) !important;
  border: 2px dashed var(--border-subtle) !important;
  border-radius: var(--radius-lg) !important;
  padding: 2rem !important;
  transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── DataFrame ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
}

/* ── Alert Boxes ───────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border-left-width: 3px !important;
}

/* ── Divider ───────────────────────────────────────────── */
hr { border-color: var(--border-subtle) !important; margin: 1.5rem 0 !important; }

/* ── Spinner ───────────────────────────────────────────── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Audio Player ──────────────────────────────────────── */
audio { border-radius: var(--radius-md) !important; width: 100%; }

/* ── Custom Components ─────────────────────────────────── */
.hero-section {
  background: linear-gradient(135deg, #080c10 0%, #0d1e2e 40%, #071a12 100%);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  padding: 3.5rem 3rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero-section::before {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(16,185,129,.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-section::after {
  content: '';
  position: absolute; bottom: -80px; left: 20%;
  width: 400px; height: 200px;
  background: radial-gradient(ellipse, rgba(14,165,233,.07) 0%, transparent 70%);
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.8rem; font-weight: 800;
  background: linear-gradient(135deg, #f0f4f8 0%, var(--accent-bright) 60%, #38bdf8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1; margin: 0 0 0.6rem;
}
.hero-subtitle {
  font-size: 1.05rem; color: var(--text-secondary);
  font-weight: 400; margin: 0 0 2rem;
  max-width: 600px;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--accent-dim); border: 1px solid var(--border-glow);
  border-radius: 50px; padding: 4px 14px;
  font-size: 0.72rem; font-weight: 600; color: var(--accent-bright);
  letter-spacing: .08em; text-transform: uppercase;
  margin-bottom: 1.5rem;
}
.hero-stats {
  display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;
}
.hero-stat { display: flex; flex-direction: column; }
.hero-stat-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.8rem; font-weight: 700; color: var(--accent-bright);
}
.hero-stat-lbl { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .06em; }

.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
  backdrop-filter: blur(12px);
  transition: transform .2s, border-color .2s, box-shadow .2s;
}
.glass-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-glow);
  box-shadow: var(--accent-glow);
}

.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.2rem; margin: 1.5rem 0; }
.feature-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  transition: transform .2s, border-color .2s, box-shadow .2s;
  position: relative; overflow: hidden;
}
.feature-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), #0ea5e9, transparent);
  opacity: 0; transition: opacity .2s;
}
.feature-card:hover { transform: translateY(-3px); border-color: var(--border-glow); box-shadow: var(--accent-glow); }
.feature-card:hover::before { opacity: 1; }
.feature-icon { font-size: 1.8rem; margin-bottom: 0.7rem; }
.feature-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem; color: var(--text-primary); margin-bottom: 0.4rem; }
.feature-desc  { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.55; }

.result-banner {
  background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(14,165,233,.08));
  border: 1px solid var(--border-glow);
  border-radius: var(--radius-xl);
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  display: flex; align-items: center; gap: 1.5rem;
}
.result-genre-icon { font-size: 3.5rem; }
.result-genre-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.5rem; font-weight: 800;
  background: linear-gradient(135deg, #f0f4f8, var(--accent-bright));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1;
}
.result-genre-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .08em; }

.section-label {
  font-size: 0.7rem; font-weight: 700; color: var(--accent);
  text-transform: uppercase; letter-spacing: .12em;
  margin-bottom: 0.5rem;
}
.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.4rem; font-weight: 700;
  color: var(--text-primary); margin: 0 0 1rem;
}

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.2rem 0; }
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 1.2rem 1.1rem;
  text-align: center;
  transition: transform .2s, border-color .2s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: var(--border-glow); }
.kpi-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--accent-bright); }
.kpi-lbl { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .06em; margin-top: 3px; }

.genre-icon-map { blues:'🎸', classical:'🎻', country:'🤠', disco:'🪩', hiphop:'🎤', jazz:'🎷', metal:'🤘', pop:'🎵', reggae:'🌴', rock:'🎸' }

.info-pill {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--blue-dim); border: 1px solid rgba(59,130,246,.25);
  border-radius: 50px; padding: 3px 10px;
  font-size: 0.72rem; font-weight: 600; color: #60a5fa;
}

.chunk-agree { color: #34d399; font-weight: 600; }
.chunk-dissent { color: #f87171; font-weight: 600; }

.about-arch-block {
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg); padding: 1.6rem;
}
.arch-row {
  display: flex; align-items: flex-start; gap: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border-subtle);
}
.arch-row:last-child { border-bottom: none; }
.arch-num {
  min-width: 28px; height: 28px;
  background: linear-gradient(135deg, var(--accent), #0ea5e9);
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700; color: #fff; margin-top: 2px;
}
.arch-content-title { font-weight: 600; font-size: 0.88rem; color: var(--text-primary); }
.arch-content-desc  { font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; line-height: 1.5; }

/* ── Sidebar Model Badge ───────────────────────────────── */
.model-active-badge {
  background: var(--accent-dim); border: 1px solid var(--border-glow);
  border-radius: var(--radius-sm); padding: 0.6rem 0.9rem;
  margin-top: 0.5rem;
}
.model-active-badge .badge-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .08em; }
.model-active-badge .badge-val   { font-size: 0.78rem; font-weight: 600; color: var(--accent-bright); margin-top: 2px; }

/* ── Workflow Stepper ──────────────────────────────────── */
.workflow-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  align-items: center;
  background: rgba(13, 17, 23, 0.75);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 0.8rem 1rem;
  margin-bottom: 1.2rem;
}
.wf-step {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.72rem;
  font-weight: 500;
  padding: 0.3rem 0.6rem;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-secondary);
  border: 1px solid rgba(255, 255, 255, 0.05);
  white-space: nowrap;
}
.wf-step.active {
  background: var(--accent-dim);
  color: var(--accent-bright);
  border-color: var(--border-glow);
  font-weight: 600;
}
.wf-step.done {
  background: rgba(56, 189, 248, 0.12);
  color: #38bdf8;
  border-color: rgba(56, 189, 248, 0.35);
  font-weight: 600;
}
.wf-arrow {
  color: var(--text-muted);
  font-size: 0.75rem;
  user-select: none;
}
</style>
"""

GENRE_ICONS = {
    'blues': '🎸', 'classical': '🎻', 'country': '🤠', 'disco': '🪩',
    'hiphop': '🎤', 'jazz': '🎷', 'metal': '🤘', 'pop': '🎵',
    'reggae': '🌴', 'rock': '🥁'
}


# ---------------------------------------------------------
# Callback Helpers for Safe Session State Navigation
# ---------------------------------------------------------
def set_nav_page(page_name="🎧 Prediction & Analysis"):
    """Callback executed before widgets instantiate to prevent StreamlitWidgetAlreadyInstantiatedError."""
    st.session_state["main_nav_select"] = page_name


def reset_analysis_callback():
    """Callback to safely clear current audio analysis state."""
    st.session_state["analysis_ready"] = False


# ---------------------------------------------------------
# Streamlit User Interface — Premium Edition
# ---------------------------------------------------------
def render_ui():
    st.set_page_config(
        page_title="SoundMind — Music Genre Intelligence",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject design system
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-brand-icon">🎵</div>
          <div class="sidebar-brand-text">
            SoundMind
            <div class="sidebar-brand-sub">Music Genre Intelligence</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        NAV_OPTIONS = ["🏠 Home", "🎧 Prediction & Analysis", "📊 Model Performance", "ℹ️ About"]
        if "main_nav_select" not in st.session_state or st.session_state["main_nav_select"] not in NAV_OPTIONS:
            st.session_state["main_nav_select"] = NAV_OPTIONS[0]

        app_mode = st.selectbox(
            "Navigate to",
            NAV_OPTIONS,
            label_visibility="collapsed",
            key="main_nav_select"
        )

        st.markdown("---")
        st.markdown('<div class="section-label">Model Configuration</div>', unsafe_allow_html=True)
        selected_model_label = st.selectbox(
            "Active Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            help="Leakage-Safe CRNN is the production model. Legacy Hybrid is available as a rollback."
        )
        active_model_path = AVAILABLE_MODELS[selected_model_label]

        is_primary = "Leakage-Safe" in selected_model_label
        badge_icon = "✅" if is_primary else "🔄"
        badge_type = "Production" if is_primary else "Rollback"
        st.markdown(f"""
        <div class="model-active-badge">
          <div class="badge-label">{badge_icon} Status</div>
          <div class="badge-val">{badge_type} Model Active</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#475569; line-height:1.7;">
          <b style="color:#94a3b8;">Architecture</b><br>
          4× Conv Block + BiLSTM + Attention<br><br>
          <b style="color:#94a3b8;">Input Shape</b><br>
          150 × 150 × 1 Mel-spectrogram<br><br>
          <b style="color:#94a3b8;">Track Accuracy</b><br>
          <span style="color:#34d399; font-weight:700;">81.33%</span> on GTZAN held-out test
        </div>
        """, unsafe_allow_html=True)

    # ── Map page name ────────────────────────────────────
    page = app_mode.split(" ", 1)[1] if " " in app_mode else app_mode

    # ====================================================
    # HOME PAGE
    # ====================================================
    if "Home" in page:
        # Hero Section
        st.markdown("""
        <div class="hero-section">
          <div class="hero-badge">🎵 AI Audio Intelligence Platform</div>
          <div class="hero-title">Music Genre<br>Classification</div>
          <div class="hero-subtitle">
            Deep audio analysis powered by a leakage-safe Hybrid CNN–BiLSTM–Attention network.
            Upload any track and get instant explainable genre predictions.
          </div>
          <div class="hero-stats">
            <div class="hero-stat">
              <span class="hero-stat-val">81.33%</span>
              <span class="hero-stat-lbl">Track Accuracy</span>
            </div>
            <div class="hero-stat">
              <span class="hero-stat-val">10</span>
              <span class="hero-stat-lbl">Music Genres</span>
            </div>
            <div class="hero-stat">
              <span class="hero-stat-val">999</span>
              <span class="hero-stat-lbl">Training Tracks</span>
            </div>
            <div class="hero-stat">
              <span class="hero-stat-val">0</span>
              <span class="hero-stat-lbl">Data Leaks</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_cta1, col_cta2 = st.columns([1.5, 3])
        with col_cta1:
            st.button(
                "🎧 Analyze Your Music Now",
                type="primary",
                key="hero_cta_btn",
                on_click=set_nav_page,
                args=("🎧 Prediction & Analysis",)
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Hero image
        image_path = os.path.join(BASE_DIR, "Images", "home.jpg")
        if os.path.exists(image_path):
            st.image(image_path)

        # Feature cards
        st.markdown('<div class="section-label" style="margin-top:2rem;">Core Capabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What This Platform Does</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-grid">
          <div class="feature-card">
            <div class="feature-icon">🛡️</div>
            <div class="feature-title">Leakage-Safe Training</div>
            <div class="feature-desc">Track-level train/val/test split enforced before chunk extraction. Zero acoustic cross-contamination between partitions.</div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">🧩</div>
            <div class="feature-title">Temporal Aggregation</div>
            <div class="feature-desc">Slices tracks into 4-second rolling windows with 2-second overlap. Aggregates chunk votes for high-confidence predictions.</div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Explainable AI (Grad-CAM)</div>
            <div class="feature-desc">Gradient-weighted Class Activation Maps highlight which time-frequency spectrogram regions drove each classification decision.</div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Interactive Dashboard</div>
            <div class="feature-desc">Waveforms, log Mel-spectrograms, chunk-by-chunk prediction trajectories, and full benchmark performance metrics.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Genres row
        st.markdown('<div class="section-label" style="margin-top:2rem;">Supported Genres</div>', unsafe_allow_html=True)
        genre_cols = st.columns(10)
        for i, (g, icon) in enumerate(GENRE_ICONS.items()):
            with genre_cols[i]:
                st.markdown(f"""
                <div style="text-align:center; padding:0.8rem 0.3rem;
                  background:rgba(17,24,32,0.85); border:1px solid rgba(255,255,255,0.06);
                  border-radius:12px; transition:all .2s;">
                  <div style="font-size:1.6rem;">{icon}</div>
                  <div style="font-size:0.65rem; color:#94a3b8; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:.05em;">{g}</div>
                </div>
                """, unsafe_allow_html=True)

        # Bottom Interactive CTA
        st.markdown("<br><br>", unsafe_allow_html=True)
        cta_c1, cta_c2, cta_c3 = st.columns([1, 1.6, 1])
        with cta_c2:
            st.button(
                "🚀  Start Audio Analysis & Try Examples ↗",
                type="primary",
                key="bottom_cta_btn",
                on_click=set_nav_page,
                args=("🎧 Prediction & Analysis",)
            )

    # ====================================================
    # PREDICTION & ANALYSIS PAGE
    # ====================================================
    elif "Prediction" in page:
        st.markdown('<div class="section-label">Audio Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction & Explainable Audio Analysis</div>', unsafe_allow_html=True)

        # Model badge
        st.markdown(f"""
        <div class="info-pill" style="margin-bottom:1.2rem;">
          🤖 &nbsp; Active Model: {selected_model_label}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Audio Source</div>', unsafe_allow_html=True)
        source_mode = st.radio(
            "Select Audio Input Method:",
            [
                "🎵 Pre-Loaded Verified Sample Tracks (Instant Test)",
                "📁 Upload Custom Audio File (MP3 / WAV)"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )

        audio_bytes = None
        audio_name = ""

        if "Sample" in source_mode:
            st.markdown('<div class="glass-card" style="padding:1.2rem 1.4rem; margin-bottom:1.2rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.83rem;color:#94a3b8;margin-bottom:0.6rem;">Choose from benchmark audio files (GTZAN distribution) for immediate evaluation:</div>', unsafe_allow_html=True)
            selected_sample_label = st.selectbox(
                "Choose Sample Track",
                options=list(SAMPLE_TRACKS.keys()),
                index=0,
                label_visibility="collapsed"
            )
            sample_path = SAMPLE_TRACKS[selected_sample_label]
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as sf_in:
                    audio_bytes = sf_in.read()
                audio_name = os.path.basename(sample_path)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass-card" style="padding:1.2rem 1.4rem; margin-bottom:1.2rem;">', unsafe_allow_html=True)
            test_audio = st.file_uploader(
                "Drop your audio file here or click to browse",
                type=["mp3", "wav"],
                help="Supports MP3 and WAV formats. Minimum 4.0 seconds duration required. Max upload size: 25 MB."
            )
            if test_audio is not None:
                if test_audio.size > 25 * 1024 * 1024:
                    st.error(f"⚠️ File size ({test_audio.size / (1024*1024):.1f} MB) exceeds the 25 MB deployment limit. Please upload an audio track under 25 MB.")
                else:
                    audio_bytes = test_audio.getvalue()
                    audio_name = test_audio.name
            st.markdown('</div>', unsafe_allow_html=True)

        # 9-Step Visual Audio Intelligence Workflow Stepper
        stage = "awaiting"
        if audio_bytes is not None and len(audio_bytes) > 0:
            stage = "analyzed" if st.session_state.get("analysis_ready", False) else "uploaded"
        st.markdown(render_workflow_stepper(stage), unsafe_allow_html=True)

        if audio_bytes is not None and len(audio_bytes) > 0:
            # Detect audio or model changes to prevent stale predictions
            sig = f"{audio_name}_{len(audio_bytes)}_{active_model_path}"
            if st.session_state.get("active_audio_sig") != sig:
                st.session_state["active_audio_sig"] = sig
                st.session_state["analysis_ready"] = False

            col_player, col_meta = st.columns([1, 1], gap="large")
            with col_player:
                st.markdown('<div class="section-label">Audio Playback</div>', unsafe_allow_html=True)
                audio_mime = "audio/wav" if audio_name.lower().endswith(".wav") else "audio/mp3"
                st.audio(audio_bytes, format=audio_mime)

            with col_meta:
                st.markdown('<div class="section-label">Track Metadata</div>', unsafe_allow_html=True)
                try:
                    meta = get_audio_metadata(io.BytesIO(audio_bytes))
                    dur_min = int(meta["duration_sec"] // 60)
                    dur_sec_val = int(meta["duration_sec"] % 60)
                    chan_desc = "Stereo" if meta["channels"] == 2 else "Mono"
                    st.markdown(f"""
                    <div class="glass-card" style="padding:1rem 1.2rem;">
                      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
                        <div><div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Track File</div>
                             <div style="font-size:0.88rem;font-weight:600;color:#f0f4f8;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{audio_name}</div></div>
                        <div><div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Duration</div>
                             <div style="font-size:0.88rem;font-weight:600;color:#f0f4f8;margin-top:2px;">{dur_min:02d}:{dur_sec_val:02d}</div></div>
                        <div><div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Sample Rate</div>
                             <div style="font-size:0.88rem;font-weight:600;color:#f0f4f8;margin-top:2px;">{meta['original_sr']:,} Hz</div></div>
                        <div><div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;">Channels</div>
                             <div style="font-size:0.88rem;font-weight:600;color:#f0f4f8;margin-top:2px;">{chan_desc} → Mono 22kHz</div></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as me:
                    st.warning(f"Could not read audio metadata: {me}")

            st.markdown("<br>", unsafe_allow_html=True)
            col_act1, col_act2 = st.columns([2.5, 1], gap="medium")
            with col_act1:
                run_btn = st.button(f"🚀  Analyse '{audio_name}' & Predict Genre", type="primary")
            with col_act2:
                st.button("🔄  Reset Analysis", key="reset_analysis_btn", on_click=reset_analysis_callback)

            if run_btn:
                with st.spinner("Extracting Mel-spectrograms, chunking audio, and running CRNN inference..."):
                    try:
                        X_test = load_and_preprocess_file(io.BytesIO(audio_bytes))
                        raw_wave, sr = load_audio_waveform(io.BytesIO(audio_bytes), target_sr=TARGET_SR)
                        model = load_trained_model(active_model_path)
                        predicted_idx, top_predictions, mean_probabilities, chunk_details = model_prediction(
                            X_test, model=model, return_chunk_details=True
                        )
                        top_genre = GENRE_LABELS[predicted_idx]

                        st.session_state.update({
                            "analysis_ready": True,
                            "X_test": X_test, "raw_wave": raw_wave, "sr": sr,
                            "predicted_idx": predicted_idx, "top_genre": top_genre,
                            "top_predictions": top_predictions,
                            "mean_probabilities": mean_probabilities,
                            "chunk_details": chunk_details,
                        })
                    except ValueError as ve:
                        st.error(f"⚠️ Audio Input Error: {ve}")
                    except Exception as e:
                        st.error(f"⚠️ Unexpected error during analysis: {e}")

            # ── Results ──────────────────────────────────
            if st.session_state.get("analysis_ready", False):
                X_test            = st.session_state["X_test"]
                raw_wave          = st.session_state["raw_wave"]
                sr                = st.session_state["sr"]
                predicted_idx     = st.session_state["predicted_idx"]
                top_genre         = st.session_state["top_genre"]
                top_predictions   = st.session_state["top_predictions"]
                mean_probabilities = st.session_state["mean_probabilities"]
                chunk_details     = st.session_state["chunk_details"]
                total_chunks      = len(chunk_details)
                top_icon          = GENRE_ICONS.get(top_genre, "🎵")

                # Result banner
                top_score = top_predictions[0]["score"] * 100
                st.markdown(f"""
                <div class="result-banner">
                  <div class="result-genre-icon">{top_icon}</div>
                  <div>
                    <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;">Predicted Genre</div>
                    <div class="result-genre-name">{top_genre.capitalize()}</div>
                    <div class="result-genre-sub">Model confidence · {top_score:.1f}% · {total_chunks} chunks analysed</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                tab1, tab2, tab3, tab4 = st.tabs([
                    "🏆 Scores & Rankings",
                    "🔍 Chunk Trajectory",
                    "📈 Visualizations",
                    "🧠 Grad-CAM XAI"
                ])

                # ── TAB 1: Scores ─────────────────────────
                with tab1:
                    st.markdown('<div class="section-label" style="margin-top:.5rem;">Top 3 Predictions</div>', unsafe_allow_html=True)
                    rank_cols = st.columns(3, gap="medium")
                    medals = ["🥇", "🥈", "🥉"]
                    for r, item in enumerate(top_predictions):
                        with rank_cols[r]:
                            pct = item["score"] * 100
                            icon = GENRE_ICONS.get(item["genre"], "🎵")
                            st.metric(
                                label=f"{medals[r]} Rank #{item['rank']} — {icon} {item['genre'].capitalize()}",
                                value=f"{pct:.1f}%",
                                delta="Top pick" if r == 0 else f"−{(top_predictions[0]['score']-item['score'])*100:.1f}% vs #1"
                            )

                    st.markdown("---")
                    st.markdown('<div class="section-label">All 10 Genre Scores</div>', unsafe_allow_html=True)
                    fig_bar = plot_genre_scores_chart(mean_probabilities, predicted_idx)
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

                    st.info(
                        "**Calibration Note:** Scores are averaged softmax activations across all 4-second chunks. "
                        "They reflect acoustic similarity to training distributions, not absolute real-world probabilities."
                    )

                # ── TAB 2: Chunk Trajectory ───────────────
                with tab2:
                    agreeing_chunks = sum(1 for c in chunk_details if c["agrees_with_overall"])
                    consensus_pct = (agreeing_chunks / total_chunks) * 100

                    st.markdown('<div class="section-label" style="margin-top:.5rem;">Temporal Consensus</div>', unsafe_allow_html=True)
                    cm1, cm2, cm3 = st.columns(3, gap="medium")
                    cm1.metric("Total Chunks", str(total_chunks))
                    cm2.metric("Consensus Chunks", f"{agreeing_chunks} / {total_chunks}")
                    cm3.metric("Consensus Rate", f"{consensus_pct:.1f}%")

                    dissenting = [c for c in chunk_details if not c["agrees_with_overall"]]
                    if dissenting:
                        st.warning(f"**{len(dissenting)} dissenting chunk(s) detected** — voted for a different genre than the overall track prediction.")
                        for d in dissenting:
                            st.markdown(
                                f"- **Chunk {d['chunk_idx']} · {d['time_window']}** → "
                                f"voted **{d['predicted_genre'].capitalize()}** ({d['confidence']*100:.1f}%) "
                                f"vs overall **{top_genre.capitalize()}**"
                            )
                        st.caption("*Dissenting chunks often occur at musical transitions: drum solos, spoken intros, or acoustic bridges.*")
                    else:
                        st.success(f"✅ **Unanimous Agreement** — all {total_chunks} chunks voted for **{top_genre.capitalize()}**!")

                    st.markdown("---")
                    st.markdown('<div class="section-label">Chunk Detail Table</div>', unsafe_allow_html=True)
                    table_rows = [
                        {
                            "Chunk #": c["chunk_idx"],
                            "Time Window": c["time_window"],
                            "Predicted Genre": c["predicted_genre"].capitalize(),
                            "Confidence": f"{c['confidence']*100:.2f}%",
                            "Status": "✅ Agrees" if c["agrees_with_overall"] else "⚠️ Dissenting"
                        }
                        for c in chunk_details
                    ]
                    st.dataframe(table_rows, use_container_width=True)

                # ── TAB 3: Visualizations ─────────────────
                with tab3:
                    st.markdown('<div class="section-label" style="margin-top:.5rem;">Waveform & Spectral Analysis</div>', unsafe_allow_html=True)

                    st.markdown("**① Full Waveform — Amplitude vs. Time**")
                    fig_wave = plot_waveform(raw_wave, sr)
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)

                    st.markdown("**② Full Log Mel-Spectrogram (dB)**")
                    fig_spec = plot_mel_spectrogram(raw_wave, sr)
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)

                    st.markdown("**③ Chunk Spectrogram Inspector** — CRNN Input Tensor (150×150)")
                    sel_chunk = st.slider(
                        "Select chunk", min_value=1, max_value=total_chunks, value=1,
                        help="Each 4-second window is resized to a 150×150 Mel-spectrogram before CRNN inference."
                    )
                    chunk_meta = chunk_details[sel_chunk - 1]
                    st.caption(
                        f"Chunk {sel_chunk} · {chunk_meta['time_window']} · "
                        f"Local prediction: **{chunk_meta['predicted_genre'].capitalize()}** "
                        f"({chunk_meta['confidence']*100:.1f}%)"
                    )
                    chunk_img = X_test[sel_chunk - 1].squeeze()
                    fig_chunk, ax_chunk = plt.subplots(figsize=(6, 4), facecolor='#0e1117')
                    ax_chunk.set_facecolor('#0e1117')
                    c_im = ax_chunk.imshow(chunk_img, aspect='auto', origin='lower', cmap='magma')
                    plt.colorbar(c_im, ax=ax_chunk, fraction=0.046, pad=0.04)
                    ax_chunk.set_title(f"Chunk {sel_chunk} · 150×150 CRNN Input", color='white', fontsize=11)
                    ax_chunk.set_xlabel("Time Frames", color='#b0b0b0', fontsize=9)
                    ax_chunk.set_ylabel("Mel Frequency Bins", color='#b0b0b0', fontsize=9)
                    ax_chunk.tick_params(colors='#b0b0b0')
                    fig_chunk.tight_layout()
                    st.pyplot(fig_chunk)
                    plt.close(fig_chunk)

                    # Preprocessing & Windowing Protocol Callout
                    st.markdown("""
                    <div class="glass-card" style="padding:1.1rem 1.3rem; margin-top:1.2rem;">
                      <div style="font-size:0.75rem; color:#10b981; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:0.4rem;">
                        ⚙️ Acoustic Preprocessing & Chunking Protocol
                      </div>
                      <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.6;">
                        • <strong>Resampling to 22,050 Hz:</strong> All input audio is resampled to 22.05 kHz mono, capturing musical frequencies up to the 11.025 kHz Nyquist ceiling while filtering high-frequency noise and optimizing computational efficiency.<br>
                        • <strong>Rolling 4-Second Windows:</strong> Audio is segmented into 4.0-second slices with a 2.0-second stride (50% temporal overlap) to guarantee notes and transitions spanning chunk boundaries are preserved.<br>
                        • <strong>Trailing Audio Treatment:</strong> Incomplete trailing audio under 4.0 seconds is intentionally discarded rather than artificially zero-padded or time-stretched, avoiding synthetic acoustic artifacts in CRNN convolutional filters.<br>
                        • <strong>Model Input Representation:</strong> Raw power Mel-spectrograms (128 mel bins, n_fft=2048, hop=512) resized to <code>(150, 150, 1)</code> float32 are fed to the model. The log Mel-spectrogram (dB) above is strictly for visual inspection.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── TAB 4: Grad-CAM ───────────────────────
                with tab4:
                    st.markdown('<div class="section-label" style="margin-top:.5rem;">Convolutional Saliency — Grad-CAM</div>', unsafe_allow_html=True)
                    st.markdown(
                        "**Grad-CAM (Gradient-weighted Class Activation Mapping)** highlights which time-frequency regions in the Mel-spectrogram "
                        "most strongly activated convolutional layer `conv_b4_2` toward a target genre classification."
                    )

                    col_gclass, col_gchunk = st.columns([1, 1], gap="medium")
                    with col_gclass:
                        target_class_name = st.selectbox(
                            "Target Genre for Saliency Map",
                            options=[g.capitalize() for g in GENRE_LABELS],
                            index=predicted_idx,
                            help="Analyze evidence for the top predicted genre or explore sensitivity toward any other genre."
                        )
                        target_class_idx = GENRE_LABELS.index(target_class_name.lower())

                    with col_gchunk:
                        g_chunk_idx = st.selectbox(
                            "Select Chunk for Saliency Analysis",
                            options=list(range(1, total_chunks + 1)),
                            index=0,
                            format_func=lambda idx: (
                                f"Chunk {idx} · {chunk_details[idx-1]['time_window']} → "
                                f"{chunk_details[idx-1]['predicted_genre'].capitalize()}"
                            )
                        )

                    with st.spinner(f"Computing Grad-CAM saliency for '{target_class_name}' on Chunk {g_chunk_idx}..."):
                        target_chunk_tensor = X_test[g_chunk_idx - 1]
                        active_model = load_trained_model(active_model_path)
                        cam_map = compute_gradcam(active_model, target_chunk_tensor, class_idx=target_class_idx)
                        time_window_str = chunk_details[g_chunk_idx - 1]['time_window']
                        fig_cam = plot_gradcam_overlay(
                            target_chunk_tensor, cam_map, target_class_name,
                            chunk_num=g_chunk_idx, time_window=time_window_str
                        )
                        st.pyplot(fig_cam)
                        plt.close(fig_cam)

                    interpretation = generate_gradcam_interpretation(
                        cam_map, target_class_name.lower(), g_chunk_idx, time_window_str
                    )

                    st.markdown(f"""
                    <div class="glass-card" style="padding:1.2rem 1.4rem; margin-top:1rem;">
                      <div style="font-size:0.75rem; color:#10b981; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:0.5rem;">
                        🧠 Grad-CAM Sensitivity Interpretation Panel
                      </div>
                      <div style="font-size:0.95rem; font-weight:600; color:#f0f4f8; margin-bottom:0.6rem;">
                        {interpretation['summary']}
                      </div>
                      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:0.8rem; margin:0.8rem 0;">
                        <div style="background:rgba(255,255,255,0.03); padding:0.6rem 0.8rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                          <div style="font-size:0.7rem; color:#94a3b8;">Low (0–1.5 kHz)</div>
                          <div style="font-size:1.15rem; font-weight:700; color:#38bdf8;">{interpretation['low_pct']:.1f}%</div>
                          <div style="font-size:0.65rem; color:#64748b;">Bass, Sub & Kicks</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.03); padding:0.6rem 0.8rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                          <div style="font-size:0.7rem; color:#94a3b8;">Mid (1.5–4.5 kHz)</div>
                          <div style="font-size:1.15rem; font-weight:700; color:#a78bfa;">{interpretation['mid_pct']:.1f}%</div>
                          <div style="font-size:0.65rem; color:#64748b;">Vocals & Instruments</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.03); padding:0.6rem 0.8rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                          <div style="font-size:0.7rem; color:#94a3b8;">High (4.5–11 kHz)</div>
                          <div style="font-size:1.15rem; font-weight:700; color:#f43f5e;">{interpretation['high_pct']:.1f}%</div>
                          <div style="font-size:0.65rem; color:#64748b;">Cymbals & Transients</div>
                        </div>
                      </div>
                      <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.6; margin-top:0.4rem;">
                        • <strong>Acoustic Insight:</strong> {interpretation['genre_note']}<br>
                        • <strong>Temporal Evolution:</strong> {interpretation['temporal_focus']}.<br>
                        • <strong>Technical Disclaimer:</strong> <em>Grad-CAM reflects convolutional feature sensitivity only. Bidirectional LSTM and Self-Attention temporal layers subsequently aggregate these spatial activations across all chunks. Saliency highlights acoustic affinity, not definitive musical causality.</em>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ====================================================
    # MODEL PERFORMANCE PAGE
    # ====================================================
    elif "Performance" in page:
        st.markdown('<div class="section-label">Verified Benchmark Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">GTZAN Held-Out Test Set Evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            "All metrics are from 150 completely unseen tracks (2,100 chunks) evaluated under "
            "strict **track-level isolation with zero data leakage**.",
            unsafe_allow_html=False
        )

        report = load_evaluation_report()
        if report is not None:
            track = report["track_level_metrics"]
            chunk_m = report["chunk_level_metrics"]

            # Premium KPI grid
            st.markdown(f"""
            <div class="kpi-grid">
              <div class="kpi-card">
                <div class="kpi-val">{track['accuracy']*100:.1f}%</div>
                <div class="kpi-lbl">Track Accuracy</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-val">{track['macro_f1']:.3f}</div>
                <div class="kpi-lbl">Macro F1</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-val">{track['macro_precision']*100:.1f}%</div>
                <div class="kpi-lbl">Macro Precision</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-val">{track['macro_recall']*100:.1f}%</div>
                <div class="kpi-lbl">Macro Recall</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-val">{chunk_m['accuracy']*100:.1f}%</div>
                <div class="kpi-lbl">Chunk Accuracy</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            col_cm, col_table = st.columns([1.1, 1], gap="large")

            with col_cm:
                st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
                st.markdown("150 held-out tracks — track-level majority vote")
                fig_cm = plot_confusion_matrix_heatmap(report)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with col_table:
                st.markdown('<div class="section-label">Per-Genre Breakdown</div>', unsafe_allow_html=True)
                per_class = report["per_class_classification_report"]
                table_data = []
                for g in report["genre_labels"]:
                    if g in per_class:
                        m = per_class[g]
                        table_data.append({
                            "Genre": f"{GENRE_ICONS.get(g,'🎵')} {g.capitalize()}",
                            "Precision": f"{m['precision']*100:.1f}%",
                            "Recall":    f"{m['recall']*100:.1f}%",
                            "F1":        f"{m['f1-score']*100:.1f}%",
                            "Support":   int(m['support'])
                        })
                st.dataframe(table_data, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-label">Methodology Notes</div>', unsafe_allow_html=True)
            st.markdown("""
            1. **Dataset** — GTZAN: 1,000 tracks × 30 s each. File `jazz.00054.wav` excluded (corrupted). 999 valid tracks used.
            2. **Split** — 70% train / 15% validation / 15% held-out test, partitioned **at track level** before any chunk extraction.
            3. **Zero Leakage** — No chunk from any test track was seen during training or validation.
            4. **Domain Shift** — Modern highly-compressed commercial productions or hybrid electronic subgenres may show higher confusion across adjacent genres (Rock ↔ Country ↔ Pop).
            """)
        else:
            st.warning("Evaluation report `leakage_safe_evaluation_report.json` not found in the application directory.")

    # ====================================================
    # ABOUT PAGE
    # ====================================================
    elif "About" in page:
        st.markdown('<div class="section-label">Project Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">About SoundMind</div>', unsafe_allow_html=True)

        col_arch, col_data = st.columns(2, gap="large")

        with col_arch:
            st.markdown('<div class="section-label" style="margin-top:.5rem;">Model Architecture</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="about-arch-block">
              <div class="arch-row">
                <div class="arch-num">1</div>
                <div><div class="arch-content-title">Mel-Spectrogram Preprocessing</div>
                     <div class="arch-content-desc">22,050 Hz, 4-second chunks, 2-second overlap. Raw power Mel-spectrogram resized to 150×150×1.</div></div>
              </div>
              <div class="arch-row">
                <div class="arch-num">2</div>
                <div><div class="arch-content-title">4× Convolutional Blocks</div>
                     <div class="arch-content-desc">32 → 64 → 128 → 256 filters. Each block: Conv2D + BatchNorm + ReLU + MaxPool + Dropout.</div></div>
              </div>
              <div class="arch-row">
                <div class="arch-num">3</div>
                <div><div class="arch-content-title">Self-Attention Layer</div>
                     <div class="arch-content-desc">Learns temporal importance weights across sequential feature representations before LSTM processing.</div></div>
              </div>
              <div class="arch-row">
                <div class="arch-num">4</div>
                <div><div class="arch-content-title">Bidirectional LSTMs</div>
                     <div class="arch-content-desc">2 BiLSTM layers (256 → 128 units) capture forward and backward musical progression and rhythm.</div></div>
              </div>
              <div class="arch-row">
                <div class="arch-num">5</div>
                <div><div class="arch-content-title">Dense Classification Head</div>
                     <div class="arch-content-desc">Dense(256) with Dropout → Softmax(10). Outputs genre probability distribution across all 10 classes.</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_data:
            st.markdown('<div class="section-label" style="margin-top:.5rem;">Dataset & Models</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="glass-card">
              <div style="margin-bottom:1rem;">
                <div class="section-label" style="margin-bottom:.3rem;">GTZAN Benchmark</div>
                <div style="font-size:0.83rem;color:#94a3b8;line-height:1.6;">
                  10 genres × 100 tracks each. 30-second WAV clips digitized at diverse recording conditions.
                  1 corrupted file excluded (<code>jazz.00054.wav</code>). <strong style="color:#f0f4f8;">999 valid tracks</strong> used.
                </div>
              </div>
              <div style="margin-bottom:1rem;">
                <div class="section-label" style="margin-bottom:.3rem;">Production Model</div>
                <div style="font-size:0.83rem;color:#94a3b8;">
                  <code>Leakage_Safe_CRNN_Model.keras</code><br>
                  Phase 3B trained. 81.33% track-level accuracy. Zero leakage verified.
                </div>
              </div>
              <div style="margin-bottom:1rem;">
                <div class="section-label" style="margin-bottom:.3rem;">Rollback Model</div>
                <div style="font-size:0.83rem;color:#94a3b8;">
                  <code>Hybrid_CRNN_Model.keras</code><br>
                  Legacy model retained in the sidebar for comparative evaluation.
                </div>
              </div>
              <div>
                <div class="section-label" style="margin-bottom:.3rem;">Source Code</div>
                <div style="font-size:0.83rem;">
                  <a href="https://github.com/MSIVAPAPARAO13/-music-genre-classification"
                     target="_blank" style="color:#34d399; text-decoration:none; font-weight:600;">
                    github.com/MSIVAPAPARAO13/-music-genre-classification ↗
                  </a>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Explainability Stack</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card">
              <div style="font-size:0.83rem;color:#94a3b8;line-height:1.7;">
                • <strong style="color:#f0f4f8;">Grad-CAM</strong> — targets <code>conv_b4_2</code> (final conv block)<br>
                • Highlights time-frequency spectrogram saliency regions per chunk<br>
                • Falls back to last Conv2D layer if named layer is absent<br>
                • Heatmap normalized to [0, 1] and overlaid at α=0.55 on raw spectrogram
              </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_ui()

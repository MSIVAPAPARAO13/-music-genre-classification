import os
import json
import hashlib
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


def load_and_preprocess_file(file_path, target_shape=TARGET_SHAPE):
    """
    Load an audio file, resample to TARGET_SR (22,050 Hz) to match training data,
    slice into 4-second chunks with 2-second overlap, and extract 150x150 Mel spectrograms.
    Incomplete chunks (< 4.0s) and empty audio are rejected to prevent model input distortion.
    """
    if hasattr(file_path, "seek"):
        file_path.seek(0)

    try:
        audio_data, sample_rate = librosa.load(file_path, sr=TARGET_SR)
    except Exception as e:
        raise ValueError(f"Failed to read audio file. Please ensure the file is a valid MP3 or WAV audio track: {e}")

    if audio_data is None or len(audio_data) == 0:
        raise ValueError("Audio file is empty or contains no decodable audio stream.")

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

    return np.array(data, dtype=np.float32)


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
    Targets layer_name (default: 'conv_b4_2' in Leakage_Safe_CRNN).
    Returns normalized (150, 150) 2D float array in range [0, 1].
    """
    # Ensure chunk_spec is shape (1, 150, 150, 1)
    if chunk_spec.ndim == 3:
        chunk_spec = np.expand_dims(chunk_spec, axis=0)
    chunk_tensor = tf.convert_to_tensor(chunk_spec, dtype=tf.float32)

    # Locate target layer
    target_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            target_layer = layer
            break

    # Fallback to last Conv2D layer if named layer not found
    if target_layer is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break

    if target_layer is None:
        return np.zeros((150, 150), dtype=np.float32)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(chunk_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((150, 150), dtype=np.float32)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activation channels
    cam = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    cam = tf.maximum(cam, 0.0)  # ReLU
    max_val = tf.math.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val

    # Resize to original (150, 150)
    cam_resized = tf.image.resize(cam[..., tf.newaxis], TARGET_SHAPE)
    return cam_resized.numpy().squeeze()


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
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    duration = len(audio_data) / sr

    im = ax.imshow(
        log_melspec, aspect='auto', origin='lower',
        cmap='magma', extent=[0, duration, 0, sr // 2]
    )
    cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#b0b0b0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#b0b0b0')
    cbar.outline.set_edgecolor('#333333')

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


def plot_gradcam_overlay(chunk_spec, cam_heatmap, genre_name, chunk_num=1):
    """Overlay Grad-CAM heatmap on single chunk spectrogram."""
    chunk_spec = np.squeeze(chunk_spec)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Display spectrogram base
    ax.imshow(chunk_spec, aspect='auto', origin='lower', cmap='gray')
    # Overlay heatmap with alpha transparency
    im = ax.imshow(cam_heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.55)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#b0b0b0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#b0b0b0')
    cbar.outline.set_edgecolor('#333333')
    cbar.set_label("Saliency Weight", color='#b0b0b0', fontsize=9)

    ax.set_title(
        f"Grad-CAM Saliency for '{genre_name.capitalize()}' — Chunk {chunk_num}",
        color='white', fontsize=11, pad=10
    )
    ax.set_xlabel("Time Frames (Resized: 150 bins)", color='#b0b0b0', fontsize=9)
    ax.set_ylabel("Mel Frequency (Resized: 150 bins)", color='#b0b0b0', fontsize=9)
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
# Streamlit User Interface
# ---------------------------------------------------------
def render_ui():
    st.set_page_config(
        page_title="Music Genre Classifier — Explainable AI Dashboard",
        page_icon="🎵",
        layout="wide"
    )

    # Consistent, lightweight visual polish shared across all pages.
    st.markdown("""
    <style>
      :root { --music-green: #1DB954; --music-bg: #0b1220; --music-panel: #121c2d; --music-border: #263449; }
      .stApp { background: radial-gradient(ellipse at top left, #15243a 0%, #0b1220 52%, #0a101b 100%); }
      .block-container { max-width: 1400px; padding-top: 1.8rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: -0.025em; }
      h1 { font-weight: 800 !important; }
      [data-testid="stSidebar"] { background: #0a111d; border-right: 1px solid var(--music-border); }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f8fafc; }
      div[data-testid="stMetric"] { background: linear-gradient(145deg, #172338, #111a29); border: 1px solid var(--music-border); border-radius: 14px; padding: 14px 16px; }
      div[data-testid="stMetric"] label { color: #aab8ca !important; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc; }
      .stButton > button { border-radius: 10px; min-height: 2.8rem; font-weight: 650; transition: transform .15s ease, border-color .15s ease; }
      .stButton > button:hover { transform: translateY(-1px); border-color: var(--music-green); }
      .stButton > button[kind="primary"] { background: var(--music-green); color: #07110a; border: 1px solid var(--music-green); }
      div[data-testid="stFileUploader"] { background: rgba(18, 28, 45, .72); border: 1px dashed #3b526e; border-radius: 16px; padding: 12px; }
      div[data-testid="stTabs"] button[role="tab"] { border-radius: 9px 9px 0 0; font-weight: 600; }
      div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom-color: var(--music-green) !important; color: var(--music-green) !important; }
      div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid var(--music-border); border-radius: 12px; overflow: hidden; }
      div[data-testid="stAlert"] { border-radius: 12px; }
      hr { border-color: var(--music-border); }
      @media (max-width: 768px) { .block-container { padding-left: 1rem; padding-right: 1rem; } h1 { font-size: 1.8rem !important; } }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🎵 Navigation")
    page_options = ["Home", "Prediction & Analysis", "Model Performance", "About"]
    requested_page = st.session_state.pop("_requested_page", None)
    page_index = page_options.index(requested_page) if requested_page in page_options else 0
    app_mode = st.sidebar.selectbox(
        "Select Page",
        page_options,
        index=page_index,
        key="app_mode"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Configuration")
    selected_model_label = st.sidebar.selectbox(
        "Active Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Select the inference model. Default is the validated leakage-safe model with rollback support."
    )
    active_model_path = AVAILABLE_MODELS[selected_model_label]

    # -----------------------------------------------------
    # HOME PAGE
    # -----------------------------------------------------
    if app_mode == "Home":
        st.markdown(
            """
            <div class="music-hero">
              <div class="music-eyebrow">DEEP LEARNING · AUDIO INTELLIGENCE</div>
              <h1>Discover the genre behind your sound.</h1>
              <p>Upload a track, explore its acoustic fingerprint, and see how a hybrid CNN–BiLSTM–Attention model classifies music across ten genres.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        hero_cols = st.columns([1.15, 0.85], gap="large")
        with hero_cols[0]:
            image_path = os.path.join(BASE_DIR, "Images", "home.jpg")
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            st.markdown(
                """
                <div class="music-panel">
                  <h3>Explore your audio</h3>
                  <p>Start with an MP3 or WAV file that is at least four seconds long. The app resamples it to 22,050 Hz, analyzes overlapping four-second windows, and combines their model scores.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hero_cols[1]:
            st.markdown("#### What you can explore")
            st.markdown(
                """
                <div class="music-feature">
                  <div class="music-feature-icon">♫</div>
                  <div><strong>Genre prediction</strong><p>Top-three rankings and scores for all ten genres.</p></div>
                </div>
                <div class="music-feature">
                  <div class="music-feature-icon">〰</div>
                  <div><strong>Audio visualization</strong><p>Waveform, mel-spectrogram, and chunk-by-chunk results.</p></div>
                </div>
                <div class="music-feature">
                  <div class="music-feature-icon">◎</div>
                  <div><strong>Model insights</strong><p>Grad-CAM visualization and held-out GTZAN evaluation metrics.</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### Genres supported")
            st.caption("Blues · Classical · Country · Disco · Hip-hop · Jazz · Metal · Pop · Reggae · Rock")
            if st.button("🎧 Open Prediction & Analysis", type="primary", use_container_width=True):
                st.session_state["_requested_page"] = "Prediction & Analysis"
                st.rerun()

        st.markdown(
            """
            <style>
              .music-hero { padding: 1.5rem 1.7rem; margin: .2rem 0 1.5rem; border: 1px solid #2b415b; border-radius: 20px; background: linear-gradient(115deg, rgba(29,185,84,.16), rgba(20,34,54,.95) 48%, rgba(24,38,59,.82)); }
              .music-eyebrow { color: #59df8a; font-size: .75rem; font-weight: 800; letter-spacing: .16em; margin-bottom: .7rem; }
              .music-hero h1 { font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.08; margin: 0 0 .8rem; color: #f8fafc; }
              .music-hero p { max-width: 760px; color: #c3d0df; font-size: 1.05rem; margin: 0; line-height: 1.65; }
              .music-panel { padding: 1rem 1.1rem; margin-top: .8rem; border: 1px solid #263449; border-radius: 14px; background: rgba(18,28,45,.78); }
              .music-panel h3 { margin: 0 0 .4rem; color: #f8fafc; }
              .music-panel p, .music-feature p { color: #b7c4d5; font-size: .93rem; line-height: 1.55; margin: .25rem 0 0; }
              .music-feature { display: flex; gap: .8rem; align-items: flex-start; padding: .9rem 0; border-bottom: 1px solid #263449; }
              .music-feature:last-child { border-bottom: 0; }
              .music-feature-icon { flex: 0 0 2.35rem; height: 2.35rem; display: grid; place-items: center; border-radius: 10px; background: rgba(29,185,84,.13); color: #59df8a; font-size: 1.35rem; }
              .music-feature strong { color: #f8fafc; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # PREDICTION & ANALYSIS PAGE
    # -----------------------------------------------------
    elif app_mode == "Prediction & Analysis":
        st.header("🎧 Prediction & Explainable Audio Analysis")
        st.caption(f"Currently Active Model: **{selected_model_label}**")

        test_audio = st.file_uploader(
            "Upload an Audio File (MP3 or WAV)", type=["mp3", "wav"],
            help="Upload a music clip at least 4 seconds long. Analysis runs locally in this app session."
        )

        # Clear old results whenever the uploaded file or active model changes. Without this,
        # Streamlit reruns could show predictions from a previous track beside a new upload.
        current_analysis_key = None
        if test_audio is not None:
            current_analysis_key = hashlib.sha256(test_audio.getvalue()).hexdigest() + "|" + active_model_path
        previous_analysis_key = st.session_state.get("_analysis_input_key")
        if current_analysis_key != previous_analysis_key:
            st.session_state["analysis_ready"] = False
            for result_key in (
                "X_test", "raw_wave", "sr", "predicted_idx", "top_genre",
                "top_predictions", "mean_probabilities", "chunk_details"
            ):
                st.session_state.pop(result_key, None)
            st.session_state["_analysis_input_key"] = current_analysis_key

        if test_audio is not None:
            # Metadata & Audio player row
            col_player, col_meta = st.columns([1, 1])
            with col_player:
                st.subheader("🔊 Audio Playback")
                st.audio(test_audio)

            with col_meta:
                st.subheader("📋 Track Information")
                try:
                    meta = get_audio_metadata(test_audio)
                    dur_min = int(meta["duration_sec"] // 60)
                    dur_sec = int(meta["duration_sec"] % 60)
                    chan_desc = "Stereo (2 Channels)" if meta["channels"] == 2 else "Mono (1 Channel)"

                    st.markdown(f"""
                    - **Format**: `{meta['format']}`
                    - **Duration**: `{dur_min:02d}:{dur_sec:02d}` ({meta['duration_sec']:.2f} seconds)
                    - **Native Sample Rate**: `{meta['original_sr']:,} Hz`
                    - **Channels**: `{chan_desc}` (resampled to `22,050 Hz Mono` for inference)
                    """)
                except Exception as me:
                    st.warning(f"Could not read detailed audio header metadata: {me}")

            st.markdown("---")

            if st.button("🚀 Run Genre Prediction & Acoustic Analysis", type="primary"):
                with st.spinner("Extracting features, chunking spectrograms, and running inference..."):
                    try:
                        # 1. Preprocess chunks for inference
                        X_test = load_and_preprocess_file(test_audio)
                        # 2. Extract full waveform for visualization
                        raw_wave, sr = load_audio_waveform(test_audio, target_sr=TARGET_SR)
                        # 3. Load active model
                        model = load_trained_model(active_model_path)
                        # 4. Run prediction with chunk breakdown
                        predicted_idx, top_predictions, mean_probabilities, chunk_details = model_prediction(
                            X_test, model=model, return_chunk_details=True
                        )
                        top_genre = GENRE_LABELS[predicted_idx]

                        # Store in session state for instant tab switching
                        st.session_state["analysis_ready"] = True
                        st.session_state["X_test"] = X_test
                        st.session_state["raw_wave"] = raw_wave
                        st.session_state["sr"] = sr
                        st.session_state["predicted_idx"] = predicted_idx
                        st.session_state["top_genre"] = top_genre
                        st.session_state["top_predictions"] = top_predictions
                        st.session_state["mean_probabilities"] = mean_probabilities
                        st.session_state["chunk_details"] = chunk_details

                    except ValueError as ve:
                        st.error(f"⚠️ Audio Input Error: {ve}")
                    except Exception as e:
                        st.error(f"⚠️ Unexpected Error during analysis: {e}")

            # Display results if available in session state
            if st.session_state.get("analysis_ready", False):
                X_test = st.session_state["X_test"]
                raw_wave = st.session_state["raw_wave"]
                sr = st.session_state["sr"]
                predicted_idx = st.session_state["predicted_idx"]
                top_genre = st.session_state["top_genre"]
                top_predictions = st.session_state["top_predictions"]
                mean_probabilities = st.session_state["mean_probabilities"]
                chunk_details = st.session_state["chunk_details"]

                # 4 Main Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🎧 Overall Prediction & Scores",
                    "🔍 Chunk-Level Trajectory",
                    "📈 Audio & Spectral Visualizations",
                    "🧠 Explainability (Grad-CAM)"
                ])

                # ---------------------------------------------
                # TAB 1: OVERALL PREDICTION & SCORES
                # ---------------------------------------------
                with tab1:
                    st.subheader(f"🎶 Final Predicted Genre: **{top_genre.capitalize()}**")

                    metric_cols = st.columns(3)
                    for r, item in enumerate(top_predictions):
                        with metric_cols[r]:
                            pct = item["score"] * 100
                            st.metric(
                                label=f"Rank #{item['rank']} Genre",
                                value=item["genre"].capitalize(),
                                delta=f"{pct:.2f}% score"
                            )

                    st.markdown("#### All 10 Genre Scores Breakdown")
                    fig_bar = plot_genre_scores_chart(mean_probabilities, predicted_idx)
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

                    st.info("""
                    **⚠️ Probability Calibration Notice**:
                    The displayed scores reflect softmax activation values averaged across all 4-second audio chunks.
                    They indicate relative acoustic similarity to training clusters rather than absolute real-world probabilities.
                    """)

                # ---------------------------------------------
                # TAB 2: CHUNK-LEVEL TRAJECTORY
                # ---------------------------------------------
                with tab2:
                    st.subheader("🔍 Chunk-by-Chunk Prediction Trajectory")
                    total_chunks = len(chunk_details)
                    agreeing_chunks = sum(1 for c in chunk_details if c["agrees_with_overall"])
                    consensus_pct = (agreeing_chunks / total_chunks) * 100

                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Total 4-second Chunks", f"{total_chunks}")
                    c_m2.metric("Consensus Chunks", f"{agreeing_chunks} / {total_chunks}")
                    c_m3.metric("Consensus Rate", f"{consensus_pct:.1f}%")

                    # Highlight dissenting chunks
                    dissenting = [c for c in chunk_details if not c["agrees_with_overall"]]
                    if len(dissenting) > 0:
                        st.warning(f"**{len(dissenting)} Dissenting Chunk(s) Detected:**")
                        for d in dissenting:
                            st.markdown(
                                f"- **Chunk {d['chunk_idx']} ({d['time_window']})**: "
                                f"voted for **{d['predicted_genre'].capitalize()}** ({d['confidence']*100:.1f}%), "
                                f"differing from overall **{top_genre.capitalize()}**."
                            )
                        st.caption("ℹ️ *Dissenting chunks commonly occur during transitional musical passages such as drum solos, spoken intros, or acoustic bridges.*")
                    else:
                        st.success(f"**Unanimous Agreement**: All {total_chunks} chunks unanimously voted for {top_genre.capitalize()}!")

                    st.markdown("#### Detailed Chunk Table")
                    table_rows = [
                        {
                            "Chunk #": c["chunk_idx"],
                            "Time Window": c["time_window"],
                            "Predicted Genre": c["predicted_genre"].capitalize(),
                            "Confidence": f"{c['confidence']*100:.2f}%",
                            "Status": "✅ Agreed" if c["agrees_with_overall"] else "⚠️ Dissenting"
                        }
                        for c in chunk_details
                    ]
                    st.dataframe(table_rows, use_container_width=True)

                # ---------------------------------------------
                # TAB 3: AUDIO & SPECTRAL VISUALIZATIONS
                # ---------------------------------------------
                with tab3:
                    st.subheader("📈 Audio Waveform & Spectral Signatures")

                    st.markdown("##### 1. Full Audio Waveform")
                    fig_wave = plot_waveform(raw_wave, sr)
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)

                    st.markdown("##### 2. Full Log Mel-Spectrogram (dB)")
                    fig_spec = plot_mel_spectrogram(raw_wave, sr)
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)

                    st.markdown("##### 3. Individual Chunk Spectrogram Inspector")
                    selected_chunk_idx = st.slider(
                        "Select Chunk to Inspect",
                        min_value=1,
                        max_value=total_chunks,
                        value=1,
                        help="Inspect the 150x150 Mel spectrogram fed into the neural network for this specific window."
                    )
                    chunk_meta = chunk_details[selected_chunk_idx - 1]
                    st.caption(f"Inspecting **Chunk {selected_chunk_idx}** ({chunk_meta['time_window']}) — Local Prediction: **{chunk_meta['predicted_genre'].capitalize()}** ({chunk_meta['confidence']*100:.1f}%)")

                    # Display chunk spectrogram image
                    chunk_img = X_test[selected_chunk_idx - 1].squeeze()
                    fig_chunk, ax_chunk = plt.subplots(figsize=(6, 4), facecolor='#0e1117')
                    ax_chunk.set_facecolor('#0e1117')
                    c_im = ax_chunk.imshow(chunk_img, aspect='auto', origin='lower', cmap='magma')
                    plt.colorbar(c_im, ax=ax_chunk, fraction=0.046, pad=0.04)
                    ax_chunk.set_title(f"Chunk {selected_chunk_idx} Model Input Tensor (150x150)", color='white', fontsize=11)
                    ax_chunk.set_xlabel("Time Frames (Resized: 150)", color='#b0b0b0', fontsize=9)
                    ax_chunk.set_ylabel("Mel Frequency Bins (Resized: 150)", color='#b0b0b0', fontsize=9)
                    ax_chunk.tick_params(colors='#b0b0b0')
                    fig_chunk.tight_layout()
                    st.pyplot(fig_chunk)
                    plt.close(fig_chunk)

                # ---------------------------------------------
                # TAB 4: EXPLAINABILITY (GRAD-CAM)
                # ---------------------------------------------
                with tab4:
                    st.subheader("🧠 Explainability via Grad-CAM (Convolutional Saliency)")
                    st.markdown("""
                    **Grad-CAM (Gradient-weighted Class Activation Mapping)** examines the final convolutional layer 
                    (`conv_b4_2`) to highlight which time-frequency regions in the Mel-spectrogram most strongly activated 
                    the model in favor of the predicted genre.
                    """)

                    g_chunk_idx = st.selectbox(
                        "Select Chunk for Saliency Analysis",
                        options=list(range(1, total_chunks + 1)),
                        index=0,
                        format_func=lambda idx: f"Chunk {idx} ({chunk_details[idx-1]['time_window']}) - Voted {chunk_details[idx-1]['predicted_genre'].capitalize()}"
                    )

                    with st.spinner("Computing gradient-weighted class activation map..."):
                        target_chunk_tensor = X_test[g_chunk_idx - 1]
                        active_model = load_trained_model(active_model_path)
                        cam_map = compute_gradcam(active_model, target_chunk_tensor, class_idx=predicted_idx)

                        fig_cam = plot_gradcam_overlay(
                            target_chunk_tensor, cam_map, top_genre, chunk_num=g_chunk_idx
                        )
                        st.pyplot(fig_cam)
                        plt.close(fig_cam)

                    st.markdown("""
                    #### 🔬 How to Interpret This Visualization:
                    - **Warm Colors (Red / Yellow)**: Time-frequency coordinates that produced strong positive gradient activations for **""" + top_genre.capitalize() + """**.
                    - **Cool Colors (Blue / Transparent)**: Areas with low or neutral influence on the genre decision.
                    - **Technical Boundary**: Grad-CAM visualizes **convolutional feature sensitivity**. It does not constitute definitive proof of musical causality, as subsequent BiLSTM and Attention layers further integrate long-term temporal context.
                    """)

    # -----------------------------------------------------
    # MODEL PERFORMANCE PAGE
    # -----------------------------------------------------
    elif app_mode == "Model Performance":
        st.header("📊 Benchmark Evaluation & Model Performance")
        st.markdown("""
        All metrics below are verified results from the **GTZAN Held-Out Test Set** (150 completely unseen tracks / 2,100 chunks).
        The evaluation was conducted under strict track-level isolation with **zero data leakage**.
        """)

        report = load_evaluation_report()
        if report is not None:
            # Top KPI metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Track Accuracy", f"{report['track_level_metrics']['accuracy']*100:.2f}%")
            m2.metric("Macro F1-Score", f"{report['track_level_metrics']['macro_f1']:.4f}")
            m3.metric("Macro Precision", f"{report['track_level_metrics']['macro_precision']*100:.2f}%")
            m4.metric("Macro Recall", f"{report['track_level_metrics']['macro_recall']*100:.2f}%")
            m5.metric("Chunk Accuracy", f"{report['chunk_level_metrics']['accuracy']*100:.2f}%")

            st.markdown("---")

            # Per-Class Table & Confusion Matrix
            col_cm, col_table = st.columns([1.1, 1])

            with col_cm:
                st.subheader("Confusion Matrix (150 Held-Out Tracks)")
                fig_cm = plot_confusion_matrix_heatmap(report)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with col_table:
                st.subheader("Per-Genre Performance Breakdown")
                per_class = report["per_class_classification_report"]
                table_data = []
                for g in report["genre_labels"]:
                    if g in per_class:
                        metrics = per_class[g]
                        table_data.append({
                            "Genre": g.capitalize(),
                            "Precision": f"{metrics['precision']*100:.1f}%",
                            "Recall": f"{metrics['recall']*100:.1f}%",
                            "F1-Score": f"{metrics['f1-score']*100:.1f}%",
                            "Support": int(metrics['support'])
                        })
                st.dataframe(table_data, use_container_width=True)

            st.markdown("---")
            st.subheader("⚠️ Benchmark Methodology & Generalization Notes")
            st.markdown("""
            1. **Benchmark Specificity**: GTZAN consists of 1,000 thirty-second clips digitized in the late 1990s and early 2000s under specific acoustic standards.
            2. **Zero Track Leakage**: Songs were partitioned (70% train / 15% val / 15% test) *before* chunk extraction. No chunk of any test track was seen during training.
            3. **Acoustic Domain Shifts**: Modern commercial music produced with heavy dynamic range compression, distinct mastering, or hybrid electronic subgenres may exhibit higher confusion rates, particularly across adjacent genres such as Rock, Country, and Pop.
            """)
        else:
            st.warning("Evaluation report `leakage_safe_evaluation_report.json` was not found.")

    # -----------------------------------------------------
    # ABOUT PAGE
    # -----------------------------------------------------
    elif app_mode == "About":
        st.header("🎵 About the Music Genre Classification Project")
        st.markdown("""
        ### Deep Learning Architecture
        This application utilizes a custom **Hybrid CNN–BiLSTM–Attention** deep learning network:
        - **Spectrogram Preprocessing**: 22,050 Hz sampling rate, 4-second rolling chunks with 2-second overlap, raw power Mel-spectrogram resized to `(150, 150, 1)`.
        - **Convolutional Feature Extractor**: 4 sequential blocks with 32, 64, 128, and 256 filters, Batch Normalization, and MaxPooling.
        - **Context Attention Layer**: Learns temporal importance weights across feature representations.
        - **Bidirectional LSTMs**: 2 BiLSTM layers (256 and 128 units) capture musical progression and rhythmic rhythm.
        - **Dense Classification Head**: Dense(256) with Dropout followed by a 10-unit Softmax output.

        ---

        ### Dataset & Rollback Availability
        - **GTZAN Benchmark**: 10 genres with 100 audio files each. Corrupted file `jazz.00054.wav` is excluded (999 valid tracks).
        - **Active Model**: `Leakage_Safe_CRNN_Model.keras` (Phase 3B trained checkpoint).
        - **Rollback Model**: `Hybrid_CRNN_Model.keras` (retained in the sidebar for comparative evaluation).
        - **Source Code**: [https://github.com/MSIVAPAPARAO13/-music-genre-classification](https://github.com/MSIVAPAPARAO13/-music-genre-classification)
        """)


if __name__ == "__main__":
    render_ui()
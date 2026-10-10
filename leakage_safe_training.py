"""
Leakage-Safe Music Genre Classification: Track-Level Partitioning & Evaluation Pipeline.

In historical iterations (CRNN_TRAIN.ipynb), 4-second audio chunks with 2-second overlap
were split randomly via train_test_split. Because adjacent chunks from the same song
share 50% identical samples and acoustic features, this caused severe data leakage.

This module enforces:
1. Track-level partitioning BEFORE any audio chunking.
2. Complete isolation: no original song appears in more than one partition.
3. Stratified splits preserving genre balance.
4. Checkpoint selection strictly on validation data.
5. Final evaluation performed once on the held-out track-level test set.
"""

import os
import time
import json
import argparse
import concurrent.futures
from typing import Dict, List, Tuple, Any

import numpy as np
import librosa
import soundfile as sf
from tensorflow.image import resize

TARGET_SR = 22050
CHUNK_DURATION = 4      # seconds
OVERLAP_DURATION = 2    # seconds
TARGET_SHAPE = (150, 150)
GENRE_LABELS = [
    'blues', 'classical', 'country', 'disco',
    'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
]


def generate_track_splits(
    genres: List[str] = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, List[str]]:
    """
    Generate reproducible, stratified track-level splits for GTZAN songs.
    Handles known corrupted file (jazz.00054.wav) by exclusion before splitting.
    """
    if genres is None:
        genres = GENRE_LABELS

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    rng = np.random.RandomState(seed)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for genre in genres:
        tracks = [
            f"{genre}.{i:05d}"
            for i in range(100)
            if not (genre == "jazz" and i == 54)  # Skip known corrupt GTZAN track
        ]
        shuffled = rng.permutation(tracks).tolist()
        n = len(shuffled)
        n_test = int(np.round(test_ratio * n))
        n_val = int(np.round(val_ratio * n))

        test_partition = shuffled[:n_test]
        val_partition = shuffled[n_test : n_test + n_val]
        train_partition = shuffled[n_test + n_val:]

        splits["test"].extend(test_partition)
        splits["val"].extend(val_partition)
        splits["train"].extend(train_partition)

    for k in splits:
        splits[k].sort()

    return splits


def verify_no_track_leakage(splits: Dict[str, List[str]]) -> bool:
    """Strict assertion check proving zero track overlap between train, val, and test."""
    s_train = set(splits["train"])
    s_val = set(splits["val"])
    s_test = set(splits["test"])

    train_val_overlap = s_train.intersection(s_val)
    train_test_overlap = s_train.intersection(s_test)
    val_test_overlap = s_val.intersection(s_test)

    if train_val_overlap:
        raise ValueError(f"Leakage detected: {len(train_val_overlap)} tracks in both train and val: {train_val_overlap}")
    if train_test_overlap:
        raise ValueError(f"Leakage detected: {len(train_test_overlap)} tracks in both train and test: {train_test_overlap}")
    if val_test_overlap:
        raise ValueError(f"Leakage detected: {len(val_test_overlap)} tracks in both val and test: {val_test_overlap}")

    return True


def save_split_manifest(splits: Dict[str, List[str]], filepath: str = "gtzan_leakage_safe_split.json"):
    """Save partition manifest to disk for reproducibility and auditing."""
    verify_no_track_leakage(splits)
    manifest = {
        "description": "GTZAN Leakage-Safe Stratified Track Splits (Song-level isolation)",
        "seed": 42,
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved partition manifest to {filepath}")


def check_dataset_availability(data_dir: str) -> bool:
    """Check if the GTZAN dataset audio directory exists on disk."""
    if not os.path.isdir(data_dir):
        return False
    for g in GENRE_LABELS:
        genre_path = os.path.join(data_dir, g)
        if os.path.isdir(genre_path) and len(os.listdir(genre_path)) > 0:
            return True
    return False


def audit_gtzan_dataset(data_dir: str, manifest_path: str = "gtzan_leakage_safe_split.json") -> Dict:
    """
    Comprehensive audit of the GTZAN dataset:
    - File counts per genre
    - Readability, duration, sample rate
    - Detection of corrupted tracks (jazz.00054.wav)
    - Verification against frozen manifest
    """
    report = {
        "data_dir": data_dir,
        "exists": os.path.isdir(data_dir),
        "genres_found": {},
        "missing_genres": [],
        "total_files_found": 0,
        "valid_tracks": 0,
        "corrupted_tracks": [],
        "unexpected_files": [],
        "jazz_00054_status": "NOT_CHECKED",
        "sample_rates": {},
        "manifest_alignment": {
            "manifest_file": manifest_path,
            "manifest_loaded": False,
            "missing_from_disk": [],
            "extra_on_disk": [],
            "all_manifest_tracks_valid": False,
            "zero_overlap_confirmed": False,
        },
    }

    manifest_splits = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            manifest_splits = manifest_data.get("splits", {})
            report["manifest_alignment"]["manifest_loaded"] = True
            try:
                verify_no_track_leakage(manifest_splits)
                report["manifest_alignment"]["zero_overlap_confirmed"] = True
            except Exception as e:
                report["manifest_alignment"]["zero_overlap_confirmed"] = False

    if not os.path.isdir(data_dir):
        report["missing_genres"] = list(GENRE_LABELS)
        if manifest_splits:
            all_manifest_tracks = (
                manifest_splits.get("train", []) +
                manifest_splits.get("val", []) +
                manifest_splits.get("test", [])
            )
            report["manifest_alignment"]["missing_from_disk"] = all_manifest_tracks
        return report

    all_disk_tracks = set()

    for genre in GENRE_LABELS:
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            report["missing_genres"].append(genre)
            report["genres_found"][genre] = 0
            continue

        files = sorted(os.listdir(genre_path))
        audio_files = [f for f in files if f.endswith(".wav")]
        unexpected = [f for f in files if not f.endswith(".wav")]
        report["genres_found"][genre] = len(audio_files)
        report["total_files_found"] += len(audio_files)
        if unexpected:
            report["unexpected_files"].extend([os.path.join(genre, f) for f in unexpected])

        for fname in audio_files:
            track_id = os.path.splitext(fname)[0]
            fpath = os.path.join(genre_path, fname)

            try:
                info = sf.info(fpath)
                sr = info.samplerate
                report["sample_rates"][sr] = report["sample_rates"].get(sr, 0) + 1
                if track_id == "jazz.00054":
                    report["jazz_00054_status"] = f"DECODABLE (sr={sr}, dur={info.duration:.2f}s)"
                else:
                    all_disk_tracks.add(track_id)
                report["valid_tracks"] += 1
            except Exception as exc:
                if track_id == "jazz.00054":
                    report["jazz_00054_status"] = f"CORRUPTED/UNREADABLE: {exc}"
                report["corrupted_tracks"].append({"track_id": track_id, "error": str(exc)})

    if manifest_splits:
        all_manifest_tracks = set(
            manifest_splits.get("train", []) +
            manifest_splits.get("val", []) +
            manifest_splits.get("test", [])
        )
        missing = sorted(list(all_manifest_tracks - all_disk_tracks))
        extra = sorted(list(all_disk_tracks - all_manifest_tracks))
        report["manifest_alignment"]["missing_from_disk"] = missing
        report["manifest_alignment"]["extra_on_disk"] = extra
        report["manifest_alignment"]["all_manifest_tracks_valid"] = (
            len(missing) == 0 and len(report["corrupted_tracks"]) == 1 and report["corrupted_tracks"][0]["track_id"] == "jazz.00054"
        )

    return report


def process_single_track(args: Tuple[str, str, int]) -> Tuple[str, int, List[np.ndarray]]:
    """Process one audio track into complete 4-second mel spectrogram chunks."""
    track_id, file_path, class_idx = args
    chunks_data = []

    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return track_id, class_idx, []

    chunk_samples = CHUNK_DURATION * sr
    overlap_samples = OVERLAP_DURATION * sr
    step_samples = chunk_samples - overlap_samples

    start = 0
    while start + chunk_samples <= len(audio):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) == chunk_samples:
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr)
            mel_resized = resize(np.expand_dims(mel, axis=-1), TARGET_SHAPE).numpy()
            chunks_data.append(mel_resized)
        start += step_samples

    return track_id, class_idx, chunks_data


def extract_and_cache_features(
    splits: Dict[str, List[str]],
    data_dir: str,
    cache_dir: str = "Data/processed_cache"
) -> Dict[str, Dict[str, Any]]:
    """
    Extract Mel spectrogram chunks for each partition independently and cache to disk.
    Ensures track isolation: every chunk inherits its parent track's partition assignment.
    """
    os.makedirs(cache_dir, exist_ok=True)
    partition_data = {}

    for partition in ["train", "val", "test"]:
        x_path = os.path.join(cache_dir, f"X_{partition}.npy")
        y_path = os.path.join(cache_dir, f"y_{partition}.npy")
        meta_path = os.path.join(cache_dir, f"meta_{partition}.json")

        if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(meta_path):
            print(f"Loading cached {partition} features from {cache_dir}...")
            X = np.load(x_path)
            y = np.load(y_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            partition_data[partition] = {"X": X, "y": y, "meta": meta}
            print(f"  {partition}: {len(X)} chunks from {len(set(meta['track_ids']))} tracks.")
            continue

        print(f"\nExtracting {partition} features ({len(splits[partition])} tracks)...")
        tasks = []
        for track_id in splits[partition]:
            genre = track_id.split(".")[0]
            class_idx = GENRE_LABELS.index(genre)
            file_path = os.path.join(data_dir, genre, f"{track_id}.wav")
            tasks.append((track_id, file_path, class_idx))

        all_chunks = []
        all_labels = []
        all_track_ids = []

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_single_track, tasks)

            for track_id, class_idx, chunks in results:
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_labels.append(class_idx)
                    all_track_ids.append(track_id)

        elapsed = time.time() - t0
        X = np.array(all_chunks, dtype=np.float32)
        # Convert labels to one-hot encoding (num_classes=10)
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(all_labels, num_classes=len(GENRE_LABELS)).astype(np.float32)

        meta = {
            "track_ids": all_track_ids,
            "partition": partition,
            "total_chunks": len(all_chunks),
            "num_tracks": len(set(all_track_ids)),
        }

        np.save(x_path, X)
        np.save(y_path, y)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        partition_data[partition] = {"X": X, "y": y, "meta": meta}
        print(f"  Extracted {len(X)} chunks in {elapsed:.2f}s. Saved to {cache_dir}.")

    return partition_data


def build_hybrid_crnn_model(input_shape=(150, 150, 1), num_classes=10):
    """
    Reconstruct the exact Hybrid CRNN architecture from CRNN_TRAIN.ipynb:
    4 Conv2D blocks + Reshape to (49, 256) + Channel Attention + 2x BiLSTM + Dense.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPool2D, BatchNormalization, Reshape,
        Dense, Multiply, Bidirectional, LSTM, Activation, Dropout
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    input_tensor = Input(shape=input_shape, name="spectrogram_input")

    # Block 1
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu', name="conv_b1_1")(input_tensor)
    x = Conv2D(32, kernel_size=3, activation='relu', name="conv_b1_2")(x)
    x = MaxPool2D(pool_size=2, strides=2, name="pool_b1")(x)
    x = BatchNormalization(name="batch_normalization")(x)

    # Block 2
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu', name="conv_b2_1")(x)
    x = Conv2D(64, kernel_size=3, activation='relu', name="conv_b2_2")(x)
    x = MaxPool2D(pool_size=2, strides=2, name="pool_b2")(x)
    x = BatchNormalization(name="batch_normalization_1")(x)

    # Block 3
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name="conv_b3_1")(x)
    x = Conv2D(128, kernel_size=3, activation='relu', name="conv_b3_2")(x)
    x = MaxPool2D(pool_size=2, strides=2, name="pool_b3")(x)
    x = BatchNormalization(name="batch_normalization_2")(x)

    # Block 4
    x = Conv2D(256, kernel_size=3, padding='same', activation='relu', name="conv_b4_1")(x)
    x = Conv2D(256, kernel_size=3, activation='relu', name="conv_b4_2")(x)
    x = MaxPool2D(pool_size=2, strides=2, name="pool_b4")(x)
    x = BatchNormalization(name="batch_normalization_3")(x)

    # Temporal Reshape: (7, 7, 256) -> (49, 256)
    seq_features = Reshape(target_shape=(49, 256), name="temporal_reshape")(x)

    # Attention
    attention = Dense(256, activation='softmax', name="attention_weights")(seq_features)
    attention_mul = Multiply(name="attention_multiply")([seq_features, attention])

    # Bi-directional LSTMs
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3), name="Bi_LSTM_1")(attention_mul)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3), name="Bi_LSTM_2")(lstm_out)

    # Classification Head
    fc = Dense(256, kernel_regularizer=l2(0.001), name="FC_1")(lstm_out)
    fc = BatchNormalization(name="batch_normalization_4")(fc)
    fc = Activation('relu', name="activation")(fc)
    fc = Dropout(0.5, name="dropout")(fc)

    predictions = Dense(num_classes, activation='softmax', name="genre_classifier")(fc)

    model = Model(inputs=input_tensor, outputs=predictions, name="Leakage_Safe_CRNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_leakage_safe_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    model_save_path: str = "Leakage_Safe_CRNN_Model.keras"
):
    """
    Train model using training partition only.
    Select checkpoint using validation partition only.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model = build_hybrid_crnn_model(input_shape=X_train[0].shape, num_classes=len(GENRE_LABELS))

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    print(f"\n--- Starting Leakage-Safe Model Training ({epochs} epochs max, batch size {batch_size}) ---")
    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_scheduler, checkpoint]
    )
    training_duration = time.time() - t0
    print(f"Training completed in {training_duration/60:.2f} minutes.")

    # Save training history
    history_path = "leakage_safe_training_hist.json"
    with open(history_path, "w", encoding="utf-8") as f:
        # Convert float32 in history to standard floats
        serializable_hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(serializable_hist, f, indent=2)
    print(f"Saved training history to {history_path}")

    return model, history.history, training_duration


def evaluate_held_out_test_set(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_meta: Dict[str, Any],
    training_duration: float,
    report_output_path: str = "leakage_safe_evaluation_report.json"
) -> Dict[str, Any]:
    """
    Evaluate the selected best model strictly on the untouched test partition.
    Reports both chunk-level and track-level aggregated metrics.
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

    print("\n--- Evaluating Held-Out Test Tracks (Zero Leakage) ---")
    chunk_preds = model.predict(X_test, verbose=0)
    chunk_pred_classes = np.argmax(chunk_preds, axis=1)
    chunk_true_classes = np.argmax(y_test, axis=1)

    chunk_acc = float(accuracy_score(chunk_true_classes, chunk_pred_classes))
    chunk_p, chunk_r, chunk_f1, _ = precision_recall_fscore_support(chunk_true_classes, chunk_pred_classes, average='macro', zero_division=0)

    # Track-level evaluation via mean probability pooling
    track_ids = test_meta["track_ids"]
    unique_tracks = sorted(list(set(track_ids)))

    track_true_classes = []
    track_pred_classes = []

    for tid in unique_tracks:
        indices = [i for i, x in enumerate(track_ids) if x == tid]
        genre = tid.split(".")[0]
        true_cls = GENRE_LABELS.index(genre)
        track_true_classes.append(true_cls)

        track_chunk_probs = chunk_preds[indices]
        mean_prob = np.mean(track_chunk_probs, axis=0)
        pred_cls = int(np.argmax(mean_prob))
        track_pred_classes.append(pred_cls)

    track_acc = float(accuracy_score(track_true_classes, track_pred_classes))
    track_p, track_r, track_f1, _ = precision_recall_fscore_support(track_true_classes, track_pred_classes, average='macro', zero_division=0)

    clf_report = classification_report(track_true_classes, track_pred_classes, target_names=GENRE_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(track_true_classes, track_pred_classes).tolist()

    report = {
        "evaluation_strategy": "Held-out Song-Level Evaluation (Zero Data Leakage)",
        "hardware": "Intel 16-Core CPU (AVX2/oneDNN)",
        "training_duration_seconds": round(training_duration, 2),
        "track_counts": {
            "test_tracks": len(unique_tracks),
            "test_chunks": len(X_test),
        },
        "track_level_metrics": {
            "accuracy": round(track_acc, 4),
            "macro_precision": round(float(track_p), 4),
            "macro_recall": round(float(track_r), 4),
            "macro_f1": round(float(track_f1), 4),
        },
        "chunk_level_metrics": {
            "accuracy": round(chunk_acc, 4),
            "macro_precision": round(float(chunk_p), 4),
            "macro_recall": round(float(chunk_r), 4),
            "macro_f1": round(float(chunk_f1), 4),
        },
        "per_class_classification_report": clf_report,
        "confusion_matrix": cm,
        "genre_labels": GENRE_LABELS,
    }

    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {report_output_path}")

    print("\n========================================================")
    print("        LEAKAGE-SAFE HELD-OUT TEST RESULTS")
    print("========================================================")
    print(f"Track-Level Test Accuracy: {track_acc * 100:.2f}%")
    print(f"Track-Level Macro F1:     {track_f1:.4f}")
    print(f"Chunk-Level Accuracy:     {chunk_acc * 100:.2f}%")
    print("--------------------------------------------------------")
    for g in GENRE_LABELS:
        metrics = clf_report[g]
        print(f"  {g:10s} | Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1-score']:.2f} | Support: {int(metrics['support'])}")
    print("========================================================")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-safe GTZAN training & evaluation pipeline")
    parser.add_argument("--data_dir", type=str, default="Data/genres_original", help="Path to GTZAN genres_original directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--skip_training", action="store_true", help="Only audit and extract features")
    args = parser.parse_args()

    print("=== Step 1 & 2: GTZAN Dataset Audit & Partition Verification ===")
    audit_report = audit_gtzan_dataset(args.data_dir)
    print(f"Dataset directory: {audit_report['data_dir']} (exists: {audit_report['exists']})")
    print(f"Total audio files found: {audit_report['total_files_found']}")
    print(f"Valid tracks: {audit_report['valid_tracks']}")
    print(f"Defective track jazz.00054 status: {audit_report['jazz_00054_status']}")
    print(f"Manifest zero-overlap confirmed: {audit_report['manifest_alignment']['zero_overlap_confirmed']}")
    print(f"Manifest 1:1 match with disk: {audit_report['manifest_alignment']['all_manifest_tracks_valid']}")

    if not audit_report["exists"] or audit_report["valid_tracks"] < 990:
        print("\n[MANDATORY STOP CONDITION TRIGGERED]: Complete usable GTZAN dataset is unavailable.")
        exit(1)

    # Load frozen splits
    with open("gtzan_leakage_safe_split.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    splits = manifest["splits"]

    # Extract & Cache features
    print("\n=== Step 3: Extracting Partition Features ===")
    partition_data = extract_and_cache_features(splits, args.data_dir)

    if args.skip_training:
        print("Skipping training as requested.")
        exit(0)

    # Train model
    X_train = partition_data["train"]["X"]
    y_train = partition_data["train"]["y"]
    X_val = partition_data["val"]["X"]
    y_val = partition_data["val"]["y"]

    model_save_path = "Leakage_Safe_CRNN_Model.keras"
    model, history, train_dur = train_leakage_safe_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=model_save_path
    )

    # Evaluate held-out test partition
    print("\n=== Step 4: Held-Out Test Evaluation ===")
    import tensorflow as tf
    best_model = tf.keras.models.load_model(model_save_path, compile=False)
    X_test = partition_data["test"]["X"]
    y_test = partition_data["test"]["y"]
    test_meta = partition_data["test"]["meta"]

    eval_report = evaluate_held_out_test_set(
        best_model, X_test, y_test, test_meta, train_dur
    )

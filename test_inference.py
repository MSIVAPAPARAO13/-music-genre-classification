import os
import io
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import MagicMock

from Music_Genre_App import (
    load_and_preprocess_file,
    model_prediction,
    TARGET_SR,
    CHUNK_DURATION,
    TARGET_SHAPE,
    GENRE_LABELS,
)


def create_synthetic_wav(duration_seconds: float, sr: int = TARGET_SR) -> io.BytesIO:
    """Helper to generate in-memory synthetic audio waveforms for testing."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sr), endpoint=False)
    # Generate 440 Hz sinusoidal waveform
    waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)
    return buf


class TestAudioPreprocessing:
    def test_preprocessing_output_shape_and_dtype(self):
        """Verify 10s audio yields exactly 4 chunks of shape (150, 150, 1) and float32 dtype."""
        buf = create_synthetic_wav(duration_seconds=10.0, sr=TARGET_SR)
        chunks = load_and_preprocess_file(buf)

        assert isinstance(chunks, np.ndarray)
        # 10s audio with 4s chunk and 2s overlap: [0-4s], [2-6s], [4-8s], [6-10s] -> 4 chunks
        assert chunks.shape == (4, TARGET_SHAPE[0], TARGET_SHAPE[1], 1)
        assert chunks.dtype == np.float32
        assert not np.isnan(chunks).any()
        assert not np.isinf(chunks).any()

    def test_resampling_from_44100_to_target_sr(self):
        """Verify that audio uploaded at 44.1 kHz is resampled and yields identical chunk count."""
        buf_44100 = create_synthetic_wav(duration_seconds=10.0, sr=44100)
        chunks = load_and_preprocess_file(buf_44100)

        assert chunks.shape == (4, TARGET_SHAPE[0], TARGET_SHAPE[1], 1)

    def test_short_audio_rejection(self):
        """Verify that audio clips under 4.0 seconds raise ValueError with clear explanation."""
        buf_short = create_synthetic_wav(duration_seconds=2.5, sr=TARGET_SR)
        with pytest.raises(ValueError, match="shorter than the required 4.0 seconds"):
            load_and_preprocess_file(buf_short)

    def test_empty_audio_rejection(self):
        """Verify that empty audio raises a ValueError."""
        buf_empty = io.BytesIO()
        sf.write(buf_empty, np.array([], dtype=np.float32), TARGET_SR, format="WAV")
        buf_empty.seek(0)

        with pytest.raises(ValueError):
            load_and_preprocess_file(buf_empty)

    def test_corrupted_file_rejection(self):
        """Verify that non-audio bytes raise a ValueError."""
        buf_corrupted = io.BytesIO(b"NOT_A_VALID_AUDIO_FILE_DATA_CORRUPTED_STREAM")
        with pytest.raises(ValueError, match="Failed to read audio file"):
            load_and_preprocess_file(buf_corrupted)

    def test_incomplete_final_chunk_not_stretched(self):
        """
        Verify that incomplete tail chunks are not stretched.
        For a 5.5s clip, only chunk [0-4s] fits. The partial [2-5.5s] chunk must be dropped,
        not stretched, yielding exactly 1 chunk.
        """
        buf = create_synthetic_wav(duration_seconds=5.5, sr=TARGET_SR)
        chunks = load_and_preprocess_file(buf)

        assert chunks.shape == (1, TARGET_SHAPE[0], TARGET_SHAPE[1], 1)


class TestModelPredictionHandling:
    def test_mock_model_prediction_and_top3_ranking(self):
        """Verify probability aggregation across chunks and descending top-3 ranking."""
        mock_model = MagicMock()
        # Simulate 3 chunks with 10 genre scores each
        # Let 'jazz' (index 5) have the highest mean score, 'blues' (index 0) 2nd, 'rock' (index 9) 3rd
        chunk_scores = np.zeros((3, 10), dtype=np.float32)
        chunk_scores[:, 5] = [0.8, 0.7, 0.9]  # Jazz: mean 0.8
        chunk_scores[:, 0] = [0.1, 0.15, 0.05]  # Blues: mean 0.1
        chunk_scores[:, 9] = [0.05, 0.08, 0.02]  # Rock: mean 0.05
        # Remaining classes have very low scores
        mock_model.predict.return_value = chunk_scores

        dummy_input = np.zeros((3, 150, 150, 1), dtype=np.float32)
        predicted_idx, top_predictions, mean_scores = model_prediction(dummy_input, model=mock_model)

        assert predicted_idx == 5
        assert GENRE_LABELS[predicted_idx] == "jazz"
        assert len(top_predictions) == 3

        # Check top 3 ranking order
        assert top_predictions[0]["genre"] == "jazz"
        assert top_predictions[0]["rank"] == 1
        assert pytest.approx(top_predictions[0]["score"], rel=1e-3) == 0.8

        assert top_predictions[1]["genre"] == "blues"
        assert top_predictions[1]["rank"] == 2
        assert pytest.approx(top_predictions[1]["score"], rel=1e-3) == 0.1

        assert top_predictions[2]["genre"] == "rock"
        assert top_predictions[2]["rank"] == 3
        assert pytest.approx(top_predictions[2]["score"], rel=1e-3) == 0.05

        # Check mean_scores contains all 10 classes
        assert len(mean_scores) == 10

    def test_default_model_is_leakage_safe(self):
        """Verify load_trained_model() defaults to Leakage_Safe_CRNN_Model.keras."""
        if not os.path.exists("Leakage_Safe_CRNN_Model.keras"):
            pytest.skip("Leakage_Safe_CRNN_Model.keras not present")

        from Music_Genre_App import load_trained_model, PRIMARY_MODEL_PATH
        model = load_trained_model()
        dummy_input = np.zeros((1, 150, 150, 1), dtype=np.float32)
        preds = model.predict(dummy_input, verbose=0)

        assert preds.shape == (1, 10)
        assert pytest.approx(float(np.sum(preds)), rel=1e-2) == 1.0
        assert model.name == "Leakage_Safe_CRNN"

    def test_rollback_model_loading_and_shape(self):
        """Verify load_trained_model(ROLLBACK_MODEL_PATH) successfully loads Hybrid_CRNN_Model.keras."""
        if not os.path.exists("Hybrid_CRNN_Model.keras"):
            pytest.skip("Hybrid_CRNN_Model.keras not present")

        from Music_Genre_App import load_trained_model, ROLLBACK_MODEL_PATH
        model = load_trained_model(ROLLBACK_MODEL_PATH)
        dummy_input = np.zeros((1, 150, 150, 1), dtype=np.float32)
        preds = model.predict(dummy_input, verbose=0)

        assert preds.shape == (1, 10)
        assert pytest.approx(float(np.sum(preds)), rel=1e-2) == 1.0
        assert model.name == "Hybrid_Custom_CRNN"

    def test_model_fallback_on_missing_primary(self):
        """Verify that requesting a nonexistent path falls back to the rollback model if available."""
        from Music_Genre_App import load_trained_model, ROLLBACK_MODEL_PATH
        if not os.path.exists(ROLLBACK_MODEL_PATH):
            pytest.skip("Rollback model not present")

        fallback_model = load_trained_model("NonExistent_Path_9999.keras")
        assert fallback_model.name == "Hybrid_Custom_CRNN"

    def test_end_to_end_wav_and_mp3_prediction(self):
        """Verify end-to-end preprocessing and prediction pipeline on valid audio."""
        if not os.path.exists("Leakage_Safe_CRNN_Model.keras"):
            pytest.skip("Leakage_Safe_CRNN_Model.keras not present")

        from Music_Genre_App import load_and_preprocess_file, model_prediction, load_trained_model

        # Test with synthetic WAV
        buf = create_synthetic_wav(duration_seconds=6.0, sr=TARGET_SR)
        chunks = load_and_preprocess_file(buf)
        model = load_trained_model()
        pred_idx, top_preds, mean_probs = model_prediction(chunks, model=model)

        assert 0 <= pred_idx < 10
        assert len(top_preds) == 3
        assert top_preds[0]["genre"] in GENRE_LABELS
        assert pytest.approx(float(np.sum(mean_probs)), rel=1e-2) == 1.0

        # Test with real MP3 if available in Test_music
        mp3_path = os.path.join("Test_music", "music_disco.mp3")
        if os.path.exists(mp3_path):
            mp3_chunks = load_and_preprocess_file(mp3_path)
            assert len(mp3_chunks) > 0
            mp3_pred_idx, mp3_top_preds, _ = model_prediction(mp3_chunks, model=model)
            assert 0 <= mp3_pred_idx < 10
            assert len(mp3_top_preds) == 3

    def test_genre_labels_consistency(self):
        """Verify exact genre ordering matches between app, training pipeline, and evaluation report."""
        import json
        from Music_Genre_App import GENRE_LABELS as APP_LABELS
        from leakage_safe_training import GENRE_LABELS as TRAIN_LABELS

        expected = [
            'blues', 'classical', 'country', 'disco',
            'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        assert APP_LABELS == expected
        assert TRAIN_LABELS == expected

        if os.path.exists("leakage_safe_evaluation_report.json"):
            with open("leakage_safe_evaluation_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            assert report["genre_labels"] == expected




class TestLeakageSafeSplitting:
    def test_track_partition_zero_overlap(self):
        """Verify that track IDs do not overlap across train, val, and test partitions."""
        from leakage_safe_training import generate_track_splits, verify_no_track_leakage
        splits = generate_track_splits(seed=42)

        # Must verify with zero exception
        assert verify_no_track_leakage(splits) is True

        s_train = set(splits["train"])
        s_val = set(splits["val"])
        s_test = set(splits["test"])

        # Explicit set intersection assertions
        assert len(s_train & s_val) == 0
        assert len(s_train & s_test) == 0
        assert len(s_val & s_test) == 0

    def test_track_partition_stratification_and_counts(self):
        """Verify each genre has expected 70/15/15 track distribution."""
        from leakage_safe_training import generate_track_splits, GENRE_LABELS
        splits = generate_track_splits(seed=42)

        assert len(splits["train"]) == 699  # 9 * 70 + 69 (jazz skips corrupted track 54)
        assert len(splits["val"]) == 150    # 10 * 15
        assert len(splits["test"]) == 150   # 10 * 15

        # Check per-genre representation
        for genre in GENRE_LABELS:
            val_genre_count = sum(1 for t in splits["val"] if t.startswith(f"{genre}."))
            test_genre_count = sum(1 for t in splits["test"] if t.startswith(f"{genre}."))
            train_genre_count = sum(1 for t in splits["train"] if t.startswith(f"{genre}."))

            assert val_genre_count == 15
            assert test_genre_count == 15
            if genre == "jazz":
                assert train_genre_count == 69  # track jazz.00054 excluded
            else:
                assert train_genre_count == 70

    def test_overlap_detection_raises_error(self):
        """Verify that verify_no_track_leakage catches simulated track leakage."""
        from leakage_safe_training import verify_no_track_leakage
        leaky_splits = {
            "train": ["blues.00001", "blues.00002"],
            "val": ["blues.00003"],
            "test": ["blues.00001"],  # Leaked from train!
        }
        with pytest.raises(ValueError, match="Leakage detected"):
            verify_no_track_leakage(leaky_splits)

    def test_saved_manifest_file_integrity(self):
        """Verify that gtzan_leakage_safe_split.json exists, is valid, and matches specifications."""
        import json
        from leakage_safe_training import GENRE_LABELS

        manifest_path = "gtzan_leakage_safe_split.json"
        assert os.path.exists(manifest_path), "gtzan_leakage_safe_split.json must exist"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "splits" in manifest
        assert "counts" in manifest
        assert manifest["counts"]["train"] == 699
        assert manifest["counts"]["val"] == 150
        assert manifest["counts"]["test"] == 150

        splits = manifest["splits"]
        all_tracks = splits["train"] + splits["val"] + splits["test"]

        # No duplicate track assignments anywhere
        assert len(all_tracks) == 999
        assert len(set(all_tracks)) == 999

        # Corrupted track jazz.00054 must NOT be in manifest
        assert "jazz.00054" not in all_tracks

        # Genre-label mapping: every track must map to a valid genre
        for track_id in all_tracks:
            genre, num = track_id.split(".")
            assert genre in GENRE_LABELS
            assert len(num) == 5
            assert num.isdigit()

    def test_dataset_audit_detects_missing_directory(self):
        """Verify audit_gtzan_dataset reports missing dataset accurately."""
        from leakage_safe_training import audit_gtzan_dataset
        report = audit_gtzan_dataset("NonExistent_Data_Dir_12345")

        assert report["exists"] is False
        assert len(report["missing_genres"]) == 10
        assert report["total_files_found"] == 0
        assert len(report["manifest_alignment"]["missing_from_disk"]) == 999


class TestExplainableDashboardAndVisualizations:
    def test_audio_metadata_extraction_mono_and_stereo(self):
        """Verify get_audio_metadata extracts duration, sample rate, and channels accurately."""
        from Music_Genre_App import get_audio_metadata

        # 1. Test Mono
        buf_mono = create_synthetic_wav(duration_seconds=5.0, sr=22050)
        meta_mono = get_audio_metadata(buf_mono)
        assert pytest.approx(meta_mono["duration_sec"], rel=1e-2) == 5.0
        assert meta_mono["original_sr"] == 22050
        assert meta_mono["channels"] == 1
        assert meta_mono["format"] == "WAV"

        # 2. Test Stereo
        t = np.linspace(0, 4.0, 4 * 44100, endpoint=False)
        left = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        right = (0.5 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        stereo_wave = np.column_stack([left, right])
        buf_stereo = io.BytesIO()
        sf.write(buf_stereo, stereo_wave, 44100, format="WAV")
        buf_stereo.seek(0)

        meta_stereo = get_audio_metadata(buf_stereo)
        assert pytest.approx(meta_stereo["duration_sec"], rel=1e-2) == 4.0
        assert meta_stereo["original_sr"] == 44100
        assert meta_stereo["channels"] == 2

    def test_waveform_and_spectrogram_plots_generation(self):
        """Verify plot_waveform and plot_mel_spectrogram produce valid Matplotlib figures and close cleanly."""
        import matplotlib.pyplot as plt
        from Music_Genre_App import plot_waveform, plot_mel_spectrogram

        y = np.random.randn(22050 * 4).astype(np.float32)
        sr = 22050

        fig_wave = plot_waveform(y, sr)
        assert isinstance(fig_wave, plt.Figure)
        plt.close(fig_wave)

        fig_spec = plot_mel_spectrogram(y, sr)
        assert isinstance(fig_spec, plt.Figure)
        plt.close(fig_spec)

    def test_chunk_prediction_trajectory_and_details(self):
        """Verify model_prediction(return_chunk_details=True) returns chunk metadata and agreement flags."""
        from Music_Genre_App import model_prediction

        mock_model = MagicMock()
        # 3 chunks with 10 classes
        chunk_scores = np.zeros((3, 10), dtype=np.float32)
        # Chunks 0 and 2 vote for jazz (index 5); Chunk 1 votes for blues (index 0)
        chunk_scores[0, 5] = 0.9
        chunk_scores[1, 0] = 0.85
        chunk_scores[2, 5] = 0.8
        mock_model.predict.return_value = chunk_scores

        dummy_input = np.zeros((3, 150, 150, 1), dtype=np.float32)
        pred_idx, top_preds, mean_probs, details = model_prediction(
            dummy_input, model=mock_model, return_chunk_details=True
        )

        assert pred_idx == 5  # Jazz wins overall
        assert len(details) == 3
        # Chunk 1 and 3 agree; Chunk 2 dissents
        assert details[0]["predicted_genre"] == "jazz"
        assert details[0]["agrees_with_overall"] is True
        assert details[1]["predicted_genre"] == "blues"
        assert details[1]["agrees_with_overall"] is False
        assert details[2]["predicted_genre"] == "jazz"
        assert details[2]["agrees_with_overall"] is True
        assert "0.0s" in details[0]["time_window"]

    def test_gradcam_computation_and_overlay(self):
        """Verify compute_gradcam generates valid (150, 150) saliency map and plot_gradcam_overlay works."""
        if not os.path.exists("Leakage_Safe_CRNN_Model.keras"):
            pytest.skip("Leakage_Safe_CRNN_Model.keras not present")

        import matplotlib.pyplot as plt
        from Music_Genre_App import load_trained_model, compute_gradcam, plot_gradcam_overlay

        model = load_trained_model()
        dummy_chunk = np.random.randn(1, 150, 150, 1).astype(np.float32)
        cam = compute_gradcam(model, dummy_chunk, class_idx=6)  # Metal

        assert isinstance(cam, np.ndarray)
        assert cam.shape == (150, 150)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0 + 1e-5

        fig_cam = plot_gradcam_overlay(dummy_chunk, cam, "metal", chunk_num=1)
        assert isinstance(fig_cam, plt.Figure)
        plt.close(fig_cam)

    def test_evaluation_report_loading_and_cm_plot(self):
        """Verify load_evaluation_report loads verified metrics and plot_confusion_matrix_heatmap renders cleanly."""
        import matplotlib.pyplot as plt
        from Music_Genre_App import load_evaluation_report, plot_confusion_matrix_heatmap

        report = load_evaluation_report()
        assert report is not None
        assert report["track_level_metrics"]["accuracy"] == 0.8133
        assert report["track_level_metrics"]["macro_f1"] == 0.8093
        assert len(report["confusion_matrix"]) == 10

        fig_cm = plot_confusion_matrix_heatmap(report)
        assert isinstance(fig_cm, plt.Figure)
        plt.close(fig_cm)




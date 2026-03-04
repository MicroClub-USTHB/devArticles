"""
╔══════════════════════════════════════════════════════════════════════╗
║             audio_preprocessing.py — Audio Pipeline                  ║
║                                                                      ║
║  Complete preprocessing toolkit for speech, music, and audio ML.     ║
║                                                                      ║
║  Pipeline steps:                                                     ║
║    1. Load      — from file or array                                 ║
║    2. Resample  — standardize sample rate                            ║
║    3. Mono      — convert stereo → mono                              ║
║    4. Trim      — remove leading/trailing silence                    ║
║    5. Normalize — amplitude normalization                            ║
║    6. Feature extraction:                                            ║
║         - Raw waveform          (simplest)                           ║
║         - MFCC                  (speech recognition standard)        ║
║         - Mel Spectrogram       (used by CNNs)                       ║
║         - Chroma                (music analysis)                     ║
║         - Spectral features     (timbre, texture)                    ║
║                                                                      ║
║  Dependencies:                                                       ║
║    pip install librosa soundfile numpy                               ║
╚══════════════════════════════════════════════════════════════════════╝

KEY CONCEPTS:
  Sample Rate (sr):  Samples per second. 22050 Hz = standard. 16000 Hz = speech.
  n_fft:             FFT window size in samples. Larger = better frequency res.
  hop_length:        Stride between FFT windows. Smaller = better time res.
  n_mels:            Number of Mel filterbanks. 40 (compact) to 128 (rich).
  MFCCs:             Mel-frequency cepstral coefficients — compact speech features.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("audio_preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#  Loading and basic operations
def load_audio(
    path: Union[str, Path],
    sr: Optional[int] = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file into a numpy array.

    Supports: .wav, .mp3, .flac, .ogg, .m4a, and more.

    Args:
        path:     Path to audio file.
        sr:       Target sample rate in Hz. None = keep original.
                  16000 = standard for speech (Google, Mozilla, Whisper).
                  22050 = librosa default, good for music.
        mono:     True = convert stereo to mono (sum channels).
        duration: Load only this many seconds. None = entire file.
        offset:   Start loading from this many seconds in.

    Returns:
        (waveform, sample_rate)
        waveform: 1D float32 array (mono) or 2D (stereo if mono=False).

    Example:
        y, sr = load_audio("speech.wav", sr=16000, mono=True)
        print(f"Duration: {len(y)/sr:.2f}s, Shape: {y.shape}")
    """
    try:
        import librosa
        y, loaded_sr = librosa.load(
            str(path), sr=sr, mono=mono,
            duration=duration, offset=offset,
        )
        logger.info(f"Loaded '{Path(path).name}': {len(y)/loaded_sr:.2f}s @ {loaded_sr}Hz")
        return y, loaded_sr
    except ImportError:
        logger.error("librosa not installed. Run: pip install librosa soundfile")
        raise


def save_audio(
    waveform: np.ndarray,
    path: Union[str, Path],
    sr: int = 22050,
) -> None:
    """
    Save a waveform array to a .wav file.

    Args:
        waveform: 1D or 2D numpy array.
        path:     Output file path (.wav recommended).
        sr:       Sample rate of the waveform.

    Example:
        save_audio(y_processed, "output/clean_speech.wav", sr=16000)
    """
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr)
    logger.info(f"Saved audio to {path}")


def audio_info(waveform: np.ndarray, sr: int) -> dict:
    """
    Return metadata about an audio signal.

    Example:
        y, sr = load_audio("song.mp3")
        info = audio_info(y, sr)
        print(info)
    """
    return {
        "duration_seconds": round(len(waveform) / sr, 3),
        "sample_rate":      sr,
        "n_samples":        len(waveform),
        "channels":         waveform.ndim,
        "dtype":            str(waveform.dtype),
        "rms_db":           round(20 * np.log10(np.sqrt(np.mean(waveform**2)) + 1e-9), 2),
        "peak_db":          round(20 * np.log10(np.max(np.abs(waveform)) + 1e-9), 2),
    }

#  Signal processing
def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Change the sample rate of a waveform.

    Why resample?
      - Standardize all audio to the same rate (required for batching).
      - Downsample to reduce compute (16kHz is enough for speech).
      - Upsample for quality (rarely needed in ML).

    Args:
        waveform:  Audio array.
        orig_sr:   Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array.

    Example:
        y_16k = resample_audio(y_44k, orig_sr=44100, target_sr=16000)
    """
    if orig_sr == target_sr:
        return waveform
    import librosa
    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    logger.debug(f"Resampled {orig_sr}Hz → {target_sr}Hz")
    return resampled


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Convert stereo (2D) to mono (1D) by averaging channels.

    Shape: (2, N) or (N, 2) → (N,)
    """
    if waveform.ndim == 1:
        return waveform
    if waveform.shape[0] == 2:
        return waveform.mean(axis=0)
    if waveform.shape[1] == 2:
        return waveform.mean(axis=1)
    return waveform.mean(axis=0)


def trim_silence(
    waveform: np.ndarray,
    sr: int,
    top_db: float = 20.0,
) -> np.ndarray:
    """
    Remove leading and trailing silence from an audio clip.

    Uses a threshold: silence is defined as anything more than
    top_db decibels below the peak.

    Args:
        waveform: Audio array.
        sr:       Sample rate.
        top_db:   Threshold in dB relative to peak (higher = more aggressive).
                  20 dB = typical. 30 dB = trim more aggressively.

    Returns:
        Trimmed audio array.

    Example:
        y_trimmed = trim_silence(y, sr, top_db=25)
        print(f"Before: {len(y)/sr:.2f}s, After: {len(y_trimmed)/sr:.2f}s")
    """
    import librosa
    yt, _ = librosa.effects.trim(waveform, top_db=top_db)
    logger.debug(f"Trimmed: {len(waveform)/sr:.2f}s → {len(yt)/sr:.2f}s")
    return yt


def normalize_amplitude(
    waveform: np.ndarray,
    method: str = "peak",
    target_db: float = -3.0,
) -> np.ndarray:
    """
    Normalize audio amplitude so all clips have consistent volume.

    Methods:
        "peak":  Scale so the maximum absolute value = 1.0
                 Simple and lossless. Best for ML inputs.

        "rms":   Scale to a target RMS level in dBFS.
                 More perceptually consistent across clips.
                 target_db=-20 dB is common for speech datasets.

        "lufs":  Broadcast standard (EBU R128). Most accurate loudness
                 normalization. Requires pyloudnorm:
                 pip install pyloudnorm

    Args:
        waveform:  Audio array.
        method:    "peak" | "rms" | "lufs"
        target_db: Target level for "rms" and "lufs" methods.

    Example:
        y_norm = normalize_amplitude(y, method="peak")
        y_rms  = normalize_amplitude(y, method="rms", target_db=-20.0)
    """
    if method == "peak":
        peak = np.max(np.abs(waveform))
        if peak > 0:
            return waveform / peak
        return waveform

    elif method == "rms":
        current_rms = np.sqrt(np.mean(waveform**2))
        if current_rms < 1e-9:
            return waveform
        target_rms = 10 ** (target_db / 20.0)
        return waveform * (target_rms / current_rms)

    elif method == "lufs":
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(22050)
            loudness = meter.integrated_loudness(waveform)
            return pyln.normalize.loudness(waveform, loudness, target_db)
        except ImportError:
            logger.warning("pyloudnorm not installed. Falling back to peak normalization.")
            return normalize_amplitude(waveform, method="peak")

    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Use 'peak', 'rms', or 'lufs'.")


def pad_or_trim(
    waveform: np.ndarray,
    target_length: int,
    pad_mode: str = "zero",
) -> np.ndarray:
    """
    Pad or trim waveform to a fixed number of samples.

    Essential for batching — neural networks need fixed-size inputs.

    Args:
        waveform:      Audio array.
        target_length: Desired length in samples.
                       Tip: target_length = sr * duration_in_seconds
        pad_mode:      How to pad if shorter than target:
                       "zero"   — silence padding (most common)
                       "repeat" — repeat the audio cyclically
                       "reflect" — reflect at boundaries

    Returns:
        Array of exactly target_length samples.

    Example:
        # Make all clips exactly 3 seconds at 16kHz
        y_fixed = pad_or_trim(y, target_length=16000 * 3)
    """
    n = len(waveform)
    if n > target_length:
        return waveform[:target_length]
    elif n < target_length:
        deficit = target_length - n
        if pad_mode == "zero":
            return np.pad(waveform, (0, deficit), mode="constant")
        elif pad_mode == "repeat":
            repeats = (target_length // n) + 1
            return np.tile(waveform, repeats)[:target_length]
        elif pad_mode == "reflect":
            return np.pad(waveform, (0, deficit), mode="reflect")
    return waveform

#  Feature extraction
def extract_mfcc(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    delta: bool = True,
    delta_delta: bool = True,
) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs).

    MFCCs are THE classic feature for speech and audio recognition.
    They compress the spectral envelope of the signal into ~13 coefficients
    per frame, mimicking how the human ear perceives sound.

    Pipeline:
      1. Short-time Fourier transform (STFT) → power spectrogram
      2. Apply Mel filterbank → Mel spectrogram
      3. Take log → log-Mel spectrogram
      4. Apply DCT (cosine transform) → MFCCs

    Args:
        waveform:    1D audio array (float32).
        sr:          Sample rate.
        n_mfcc:      Number of MFCC coefficients (13 = standard, 40 = richer).
        n_fft:       FFT window size. Larger = finer frequency resolution.
        hop_length:  Frames hop. Smaller = finer time resolution.
        n_mels:      Mel filterbank bands. 128 = standard.
        delta:       Include first-order delta (velocity) features.
        delta_delta: Include second-order delta (acceleration) features.

    Returns:
        MFCC matrix of shape (n_mfcc * [1+delta+delta_delta], n_frames).
        Typical: (13, T) or (39, T) with deltas.

    Example:
        y, sr = load_audio("speech.wav", sr=16000)
        mfcc = extract_mfcc(y, sr, n_mfcc=13, delta=True, delta_delta=True)
        # mfcc.shape = (39, ~100)  ← 39 = 13 + 13 Δ + 13 ΔΔ
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr,
        n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    )

    features = [mfcc]
    if delta:
        features.append(librosa.feature.delta(mfcc, order=1))
    if delta_delta:
        features.append(librosa.feature.delta(mfcc, order=2))

    result = np.vstack(features)
    logger.debug(f"MFCC shape: {result.shape}")
    return result


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    to_db: bool = True,
) -> np.ndarray:
    """
    Compute a Mel Spectrogram.

    The Mel spectrogram is a 2D time-frequency representation that treats
    the audio like an IMAGE — which is why it's so popular for CNN-based
    audio classification (you can literally feed it to ResNet, EfficientNet, etc.)

    Args:
        waveform:    1D audio array.
        sr:          Sample rate.
        n_fft:       FFT window size (larger = better freq resolution).
        hop_length:  Step between frames (smaller = better time resolution).
        n_mels:      Number of Mel bins (rows). 128 = standard.
        fmin:        Lowest frequency (Hz).
        fmax:        Highest frequency. None = sr/2 (Nyquist limit).
        power:       1.0 = magnitude, 2.0 = power (recommended).
        to_db:       Convert to dB scale (much better dynamic range).

    Returns:
        Mel spectrogram array shape: (n_mels, n_frames).
        Typical: (128, ~100) for 2-second clip.

    Example:
        y, sr = load_audio("music.mp3")
        mel = extract_mel_spectrogram(y, sr, n_mels=128)
        # mel.shape = (128, T) — treat as an image!

        # For CNN input: expand dims and normalize
        mel_img = (mel + 80) / 80   # rough normalization to [0, 1]
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
        power=power,
    )
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    logger.debug(f"Mel spectrogram shape: {mel.shape}")
    return mel


def extract_chroma(
    waveform: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> np.ndarray:
    """
    Extract Chroma features (pitch class profile).

    Chroma vectors represent the energy in each of the 12 pitch classes
    (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Highly useful for:
      - Music chord recognition
      - Key detection
      - Music similarity / cover song detection

    Returns:
        Chroma matrix shape: (n_chroma=12, n_frames).

    Example:
        y, sr = load_audio("song.mp3")
        chroma = extract_chroma(y, sr)
        # Average over time to get a single chord profile:
        chord_profile = chroma.mean(axis=1)
    """
    import librosa
    return librosa.feature.chroma_stft(
        y=waveform, sr=sr,
        n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma,
    )


def extract_spectral_features(
    waveform: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, np.ndarray]:
    """
    Extract a comprehensive set of spectral features.

    Features extracted:
        spectral_centroid:   "Center of mass" of spectrum. Brightness indicator.
        spectral_bandwidth:  Width of spectrum around centroid. Richness.
        spectral_rolloff:    Frequency below which 85% of energy is contained.
        spectral_flatness:   How "noise-like" vs "tonal" the signal is.
        zero_crossing_rate:  How often the waveform crosses zero. High = noisy.

    These aggregate well (take mean/std over time) to create fixed-size
    feature vectors for genre classification, quality assessment, etc.

    Returns:
        Dict of feature_name → array of shape (1, n_frames).

    Example:
        features = extract_spectral_features(y, sr)
        # Create a single feature vector:
        summary = {k: v.mean() for k, v in features.items()}
        print(summary)
    """
    import librosa
    stft = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
    return {
        "spectral_centroid":   librosa.feature.spectral_centroid(S=stft, sr=sr),
        "spectral_bandwidth":  librosa.feature.spectral_bandwidth(S=stft, sr=sr),
        "spectral_rolloff":    librosa.feature.spectral_rolloff(S=stft, sr=sr),
        "spectral_flatness":   librosa.feature.spectral_flatness(S=stft),
        "zero_crossing_rate":  librosa.feature.zero_crossing_rate(waveform,
                               frame_length=n_fft, hop_length=hop_length),
    }


def summarize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Summarize a time-varying feature matrix into a fixed-size vector.

    Takes a (n_features, n_frames) matrix and computes statistics
    over the time dimension to produce a (n_features * 4,) vector.

    Statistics: mean, std, min, max

    Why? Neural networks need fixed-size inputs, but audio clips have
    variable durations. Summarizing over time solves this elegantly.

    Args:
        feature_matrix: Shape (n_features, n_frames).

    Returns:
        1D vector of shape (n_features * 4,).

    Example:
        mfcc = extract_mfcc(y, sr)          # (39, T)
        vec = summarize_features(mfcc)       # (156,) — fixed size!
    """
    return np.concatenate([
        feature_matrix.mean(axis=1),
        feature_matrix.std(axis=1),
        feature_matrix.min(axis=1),
        feature_matrix.max(axis=1),
    ])

# 🔧 PIPELINE — Compose everything
class AudioPreprocessingPipeline:
    """
    Configurable, composable audio preprocessing pipeline.

    Applies steps in order:
      load → mono → resample → trim → normalize → pad/trim → features

    Example (speech recognition features):
        pipeline = AudioPreprocessingPipeline(
            sr=16000,
            features="mfcc",
            n_mfcc=13,
            delta=True,
            fixed_duration=3.0,
        )
        feat = pipeline.transform("speech.wav")
        # feat.shape = (39, 94)  — MFCCs + deltas, 94 frames

    Example (CNN-ready mel spectrogram):
        pipeline = AudioPreprocessingPipeline(
            sr=22050,
            features="mel",
            n_mels=128,
            fixed_duration=5.0,
        )
        mel = pipeline.transform("music.mp3")
        # mel.shape = (128, 216)  — feed to 2D CNN

    Example (compact fixed-size feature vector):
        pipeline = AudioPreprocessingPipeline(
            sr=22050,
            features="mfcc",
            summarize=True,
        )
        vec = pipeline.transform("audio.wav")
        # vec.shape = (52,) — fixed size regardless of audio length
    """

    def __init__(
        self,
        sr: int = 22050,
        mono: bool = True,
        trim_silence: bool = True,
        trim_top_db: float = 20.0,
        normalize_method: str = "peak",
        fixed_duration: Optional[float] = None,
        features: str = "mfcc",          # "mfcc" | "mel" | "chroma" | "spectral" | "raw"
        # MFCC params
        n_mfcc: int = 13,
        delta: bool = True,
        delta_delta: bool = True,
        # Mel params
        n_mels: int = 128,
        # Shared
        n_fft: int = 2048,
        hop_length: int = 512,
        # Output
        summarize: bool = False,
    ):
        self.sr              = sr
        self.mono            = mono
        self.trim_silence    = trim_silence
        self.trim_top_db     = trim_top_db
        self.normalize_method = normalize_method
        self.fixed_duration  = fixed_duration
        self.features        = features
        self.n_mfcc          = n_mfcc
        self.delta           = delta
        self.delta_delta     = delta_delta
        self.n_mels          = n_mels
        self.n_fft           = n_fft
        self.hop_length      = hop_length
        self.summarize       = summarize

    def transform(self, source: Union[str, Path, tuple[np.ndarray, int]]) -> np.ndarray:
        """
        Process one audio file or (waveform, sr) tuple.

        Args:
            source: File path or (waveform, sample_rate) tuple.

        Returns:
            Feature matrix or summary vector.
        """
        # Step 1: Load
        if isinstance(source, (str, Path)):
            y, orig_sr = load_audio(source, sr=None, mono=False)
        else:
            y, orig_sr = source

        # Step 2: Mono
        if self.mono:
            y = to_mono(y)

        # Step 3: Resample
        if orig_sr != self.sr:
            y = resample_audio(y, orig_sr, self.sr)

        # Step 4: Trim silence
        if self.trim_silence:
            y = trim_silence(y, self.sr, top_db=self.trim_top_db)

        # Step 5: Normalize amplitude
        y = normalize_amplitude(y, method=self.normalize_method)

        # Step 6: Fixed duration
        if self.fixed_duration is not None:
            target_n = int(self.fixed_duration * self.sr)
            y = pad_or_trim(y, target_n)

        # Step 7: Extract features
        if self.features == "mfcc":
            feat = extract_mfcc(
                y, self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                delta=self.delta,
                delta_delta=self.delta_delta,
            )
        elif self.features == "mel":
            feat = extract_mel_spectrogram(
                y, self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
        elif self.features == "chroma":
            feat = extract_chroma(y, self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        elif self.features == "spectral":
            feats = extract_spectral_features(y, self.sr, self.n_fft, self.hop_length)
            feat = np.vstack(list(feats.values()))
        elif self.features == "raw":
            feat = y.reshape(1, -1)
        else:
            raise ValueError(f"Unknown features: {self.features}")

        # Step 8: Summarize (optional)
        if self.summarize:
            feat = summarize_features(feat)

        return feat

    def transform_batch(
        self,
        sources: list,
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        """
        Process a list of audio sources.

        Returns a list (not stacked) because audio files may have
        different lengths before padding.

        Example:
            paths = list(Path("audio/").glob("*.wav"))
            features = pipeline.transform_batch(paths)
        """
        results = []
        for i, source in enumerate(sources):
            try:
                results.append(self.transform(source))
            except Exception as e:
                logger.error(f"Failed on item {i}: {e}")
                continue
            if show_progress and i > 0 and i % 50 == 0:
                logger.info(f"Processed {i}/{len(sources)} audio files…")
        logger.info(f"Batch complete: {len(results)}/{len(sources)} succeeded")
        return results

#  QUICK-START DEMO (uses synthetic audio — no files needed)
if __name__ == "__main__":
    print("=" * 60)
    print("  audio_preprocessing.py — Demo (synthetic audio)")
    print("=" * 60)

    SR = 22050

    # Create a synthetic test signal: 440 Hz sine wave (concert A)
    # + some harmonic content + noise, 3 seconds long
    t = np.linspace(0, 3, SR * 3)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)    # 440 Hz fundamental
       + 0.25 * np.sin(2 * np.pi * 880 * t)   # 880 Hz harmonic
       + 0.1 * np.random.randn(len(t)))         # noise
    y = y.astype(np.float32)
    print(f"\n Synthetic signal: {len(y)/SR:.1f}s at {SR}Hz")
    print(f"   {audio_info(y, SR)}")

    # Basic operations
    print("\n Basic operations:")
    y_norm = normalize_amplitude(y, method="peak")
    print(f"   Peak normalized: max = {np.max(np.abs(y_norm)):.3f}")

    y_padded = pad_or_trim(y, target_length=SR * 5)
    print(f"   Padded to 5s: {len(y_padded)/SR:.1f}s")

    y_trimmed = pad_or_trim(y, target_length=SR * 2)
    print(f"   Trimmed to 2s: {len(y_trimmed)/SR:.1f}s")

    # Feature extraction
    try:
        import librosa

        print("\n Feature extraction:")

        mfcc = extract_mfcc(y, SR, n_mfcc=13, delta=True, delta_delta=True)
        print(f"   MFCC (+ Δ + ΔΔ): {mfcc.shape}  — ({mfcc.shape[0]} coefficients, {mfcc.shape[1]} frames)")

        mel = extract_mel_spectrogram(y, SR, n_mels=128)
        print(f"   Mel Spectrogram:  {mel.shape}  — (128 mel bins, {mel.shape[1]} frames)")
        print(f"   Value range: [{mel.min():.1f}, {mel.max():.1f}] dB")

        chroma = extract_chroma(y, SR)
        print(f"   Chroma:           {chroma.shape}  — 12 pitch classes")

        vec = summarize_features(mfcc)
        print(f"   MFCC summary vec: {vec.shape}  — (39 × 4 stats = 156-dim fixed vector)")

        # Full pipeline
        print("\n Full pipeline:")
        pipeline = AudioPreprocessingPipeline(
            sr=SR, features="mfcc", n_mfcc=13,
            delta=True, delta_delta=True,
            fixed_duration=3.0, summarize=False,
        )
        feat = pipeline.transform((y, SR))
        print(f"   Pipeline output: {feat.shape}")

        # Compact mode for ML
        pipeline_compact = AudioPreprocessingPipeline(
            sr=SR, features="mfcc", n_mfcc=13,
            delta=True, delta_delta=True,
            fixed_duration=3.0, summarize=True,
        )
        compact = pipeline_compact.transform((y, SR))
        print(f"   Compact vector:  {compact.shape} ← fixed size, plug directly into sklearn/torch")

    except ImportError:
        print("\n   librosa not installed. Run: pip install librosa soundfile")
        print("   (Signal operations above work without librosa)")

    print("\n Audio preprocessing demo complete.")
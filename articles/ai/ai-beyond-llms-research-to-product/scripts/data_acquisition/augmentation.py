"""
╔══════════════════════════════════════════════════════════════════════╗
║                 augmentation.py — Data Augmentation                  ║
║                                                                      ║
║  Expand your dataset without collecting new data.                    ║
║  Augmentation = programmatic label-preserving transformations.       ║
║                                                                      ║
║  Modalities:                                                         ║
║       Text  — synonym swap, deletion, insertion, back-translation    ║
║        Image — flip, rotate, crop, brightness, noise, cutout         ║
║        Audio — pitch shift, speed, noise, time-mask                  ║
║       Tabular — SMOTE-style oversampling, Gaussian noise             ║
║                                                                      ║
║  Dependencies (install only what you need):                          ║
║    pip install numpy pandas pillow                                   ║
║    pip install nltk                     # text augmentation          ║
║    pip install librosa soundfile        # audio augmentation         ║
╚══════════════════════════════════════════════════════════════════════╝

KEY CONCEPT:  Augmentation ≠ fabrication.
              Every augmented sample must preserve the original label.
              A horizontally flipped cat is still a cat.
              A sentence with one synonym swap still has the same sentiment.
"""

import io
import logging
import random
import copy
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

logger = logging.getLogger("augmentation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#  SHARED UTILITIES
class AugmentationPipeline:
    """
    Chain multiple augmentation functions into a single transform.

    Each transform is applied sequentially. You can mix modalities
    or randomize application with AugmentationPipeline.random().

    Args:
        transforms:  List of callables, each transforms an input → output.
        p:           Probability of applying the ENTIRE pipeline (default 1.0).

    Example (images):
        pipeline = AugmentationPipeline([
            lambda img: random_flip(img),
            lambda img: random_brightness(img, 0.8, 1.2),
            lambda img: add_gaussian_noise(img, std=0.02),
        ])
        aug_img = pipeline(original_img)

    Example (text):
        pipeline = AugmentationPipeline([
            synonym_replacement,
            random_deletion,
        ], p=0.8)
        aug_text = pipeline(original_text)
    """

    def __init__(self, transforms: list[Callable], p: float = 1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data
        for t in self.transforms:
            data = t(data)
        return data

    def augment_batch(self, items: list, n_augmented: int = 1) -> list:
        """
        Apply pipeline to each item in a list, generating n_augmented copies.

        Args:
            items:       Original items.
            n_augmented: Augmented copies per item.

        Returns:
            Original items + augmented copies.

        Example:
            texts = ["I love this product!", "Terrible experience."]
            all_texts = pipeline.augment_batch(texts, n_augmented=3)
            # Returns 2 originals + 6 augmented = 8 total
        """
        augmented = list(items)
        for item in items:
            for _ in range(n_augmented):
                augmented.append(self(copy.deepcopy(item)))
        return augmented

    @staticmethod
    def random(transforms: list[Callable], n: int = 2, p: float = 0.5) -> "AugmentationPipeline":
        """
        Create a pipeline that randomly selects n transforms each time.

        Args:
            transforms: Pool of possible transforms.
            n:          How many to pick each call.
            p:          Probability of applying each selected transform.

        Example:
            pipeline = AugmentationPipeline.random(
                [random_flip, random_crop, add_noise], n=2
            )
        """
        def _random_pipeline(data):
            chosen = random.sample(transforms, min(n, len(transforms)))
            for t in chosen:
                if random.random() < p:
                    data = t(data)
            return data
        return AugmentationPipeline([_random_pipeline])

#  TEXT AUGMENTATION
def synonym_replacement(text: str, p: float = 0.15) -> str:
    """
    Replace random words with synonyms from WordNet.

    One of the most common NLP augmentation techniques. Preserves
    meaning while generating lexical variety.

    Args:
        text: Input sentence.
        p:    Probability of replacing each word.

    Returns:
        Augmented text.

    Example:
        >>> synonym_replacement("The movie was amazing and beautiful")
        'The film was astonishing and beautiful'

    Note: Requires nltk wordnet:
        import nltk; nltk.download("wordnet"); nltk.download("omw-1.4")
    """
    try:
        from nltk.corpus import wordnet
    except ImportError:
        logger.warning("nltk not installed. Run: pip install nltk")
        return text

    words = text.split()
    augmented = []
    for word in words:
        if random.random() < p:
            synsets = wordnet.synsets(word.lower())
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower():
                        synonyms.add(candidate)
            if synonyms:
                augmented.append(random.choice(list(synonyms)))
                continue
        augmented.append(word)
    return " ".join(augmented)


def random_deletion(text: str, p: float = 0.10) -> str:
    """
    Randomly delete words from the text.

    Models should learn to handle partial / noisy text. This forces
    robustness. Don't delete too aggressively — p ≤ 0.15 is typical.

    Args:
        text: Input sentence.
        p:    Probability of deleting each word.

    Example:
        >>> random_deletion("The quick brown fox jumps over the lazy dog", p=0.2)
        'The quick brown jumps over lazy dog'
    """
    words = text.split()
    if len(words) == 1:
        return text
    result = [w for w in words if random.random() > p]
    return " ".join(result) if result else random.choice(words)


def random_insertion(text: str, p: float = 0.10) -> str:
    """
    Insert synonyms of existing words at random positions.

    Adds vocabulary variety while keeping the original words.

    Args:
        text: Input sentence.
        p:    Expected fraction of words to insert (governs n_insertions).

    Example:
        >>> random_insertion("The cat sat on the mat")
        'The feline cat sat on the mat'
    """
    try:
        from nltk.corpus import wordnet
    except ImportError:
        logger.warning("nltk not installed.")
        return text

    words = text.split()
    n_insertions = max(1, int(len(words) * p))

    for _ in range(n_insertions):
        anchor = random.choice(words)
        synsets = wordnet.synsets(anchor.lower())
        synonyms = []
        for syn in synsets:
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != anchor.lower():
                    synonyms.append(candidate)
        if synonyms:
            position = random.randint(0, len(words))
            words.insert(position, random.choice(synonyms))

    return " ".join(words)


def random_swap(text: str, n: int = 1) -> str:
    """
    Randomly swap two words n times.

    Lightweight augmentation that doesn't need any external resources.

    Args:
        text: Input sentence.
        n:    Number of swaps.

    Example:
        >>> random_swap("I really love this product")
        'I really product this love'
    """
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def keyboard_noise(text: str, p: float = 0.03) -> str:
    """
    Simulate realistic keyboard typos (adjacent-key errors).

    Useful for training spell-check models or testing robustness
    to OCR / user-input noise.

    Args:
        text: Input text.
        p:    Probability of corrupting each character.

    Example:
        >>> keyboard_noise("hello world", p=0.1)
        'heklo world'
    """
    # QWERTY adjacency map (simplified)
    adjacency = {
        'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wrsdf',
        'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'uojk', 'j': 'huikmn',
        'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'aqwedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx',
    }
    result = []
    for ch in text:
        if random.random() < p and ch.lower() in adjacency:
            neighbour = random.choice(adjacency[ch.lower()])
            result.append(neighbour.upper() if ch.isupper() else neighbour)
        else:
            result.append(ch)
    return "".join(result)


def augment_text_batch(
    texts: list[str],
    labels: Optional[list] = None,
    n_augmented: int = 2,
    methods: Optional[list[str]] = None,
) -> tuple[list[str], Optional[list]]:
    """
    Augment a collection of texts, optionally preserving labels.

    Args:
        texts:       List of input strings.
        labels:      Corresponding labels (preserved for each augmented copy).
        n_augmented: Number of augmented copies per text.
        methods:     Subset of ["synonym", "delete", "insert", "swap", "noise"].
                     If None, randomly applies all methods.

    Returns:
        (all_texts, all_labels) — originals + augmented.

    Example:
        X_aug, y_aug = augment_text_batch(
            texts=["Good movie", "Terrible film"],
            labels=[1, 0],
            n_augmented=3,
        )
        print(len(X_aug))  # 2 + 2×3 = 8
    """
    all_methods = {
        "synonym": synonym_replacement,
        "delete":  random_deletion,
        "insert":  random_insertion,
        "swap":    random_swap,
        "noise":   keyboard_noise,
    }
    active = {k: v for k, v in all_methods.items()
              if methods is None or k in methods}

    aug_texts  = list(texts)
    aug_labels = list(labels) if labels else None

    for i, text in enumerate(texts):
        for _ in range(n_augmented):
            method = random.choice(list(active.values()))
            aug_texts.append(method(text))
            if aug_labels is not None:
                aug_labels.append(labels[i])

    logger.info(f"Text augmentation: {len(texts)} → {len(aug_texts)} samples")
    return aug_texts, aug_labels

#  IMAGE AUGMENTATION
def random_flip(image: "Image") -> "Image":
    """Randomly flip image horizontally (50% chance)."""
    from PIL import Image as PILImage
    if random.random() > 0.5:
        return image.transpose(PILImage.FLIP_LEFT_RIGHT)
    return image


def random_rotate(image: "Image", max_angle: float = 20.0) -> "Image":
    """
    Rotate image by a random angle within ±max_angle degrees.

    Args:
        image:     PIL Image.
        max_angle: Maximum rotation in degrees.

    Example:
        img = Image.open("cat.jpg")
        aug = random_rotate(img, max_angle=15)
    """
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))


def random_crop(image: "Image", crop_fraction: float = 0.85) -> "Image":
    """
    Crop a random sub-region and resize back to original dimensions.

    Teaches the model to focus on any region, not just the center.

    Args:
        image:         PIL Image.
        crop_fraction: Fraction of original size to keep (0.7–0.95 typical).
    """
    w, h = image.size
    new_w = int(w * crop_fraction)
    new_h = int(h * crop_fraction)
    left  = random.randint(0, w - new_w)
    upper = random.randint(0, h - new_h)
    cropped = image.crop((left, upper, left + new_w, upper + new_h))
    return cropped.resize((w, h))


def random_brightness(image: "Image", low: float = 0.7, high: float = 1.3) -> "Image":
    """
    Randomly adjust brightness by a factor sampled from [low, high].

    Args:
        image: PIL Image.
        low:   Minimum brightness multiplier (< 1 = darker).
        high:  Maximum brightness multiplier (> 1 = brighter).
    """
    from PIL import ImageEnhance
    factor = random.uniform(low, high)
    return ImageEnhance.Brightness(image).enhance(factor)


def random_contrast(image: "Image", low: float = 0.7, high: float = 1.3) -> "Image":
    """Randomly adjust contrast. Factor < 1 = flat, > 1 = punchy."""
    from PIL import ImageEnhance
    factor = random.uniform(low, high)
    return ImageEnhance.Contrast(image).enhance(factor)


def add_gaussian_noise(image: "Image", std: float = 0.05) -> "Image":
    """
    Add pixel-level Gaussian noise to an image.

    Simulates sensor noise, compression artifacts, or adverse capture conditions.

    Args:
        image: PIL Image (RGB or grayscale).
        std:   Noise intensity as fraction of max value (0.02–0.10 typical).
    """
    from PIL import Image as PILImage
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return PILImage.fromarray((noisy * 255).astype(np.uint8))


def cutout(image: "Image", n_holes: int = 1, hole_size: float = 0.2) -> "Image":
    """
    Randomly mask rectangular regions with grey (CutOut regularization).

    Reference: "Improved Regularization of CNNs with Cutout" (DeVries & Taylor, 2017).
    Forces the model to look at the whole image, not just one discriminative patch.

    Args:
        image:     PIL Image.
        n_holes:   Number of rectangles to cut out.
        hole_size: Size of each hole as fraction of image size.

    Example:
        aug = cutout(img, n_holes=2, hole_size=0.15)
    """
    from PIL import ImageDraw
    aug = image.copy()
    draw = ImageDraw.Draw(aug)
    w, h = image.size
    hw = int(w * hole_size)
    hh = int(h * hole_size)
    for _ in range(n_holes):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1, y1 = max(0, cx - hw // 2), max(0, cy - hh // 2)
        x2, y2 = min(w, cx + hw // 2), min(h, cy + hh // 2)
        draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128))
    return aug


def build_image_pipeline(strength: str = "medium") -> AugmentationPipeline:
    """
    Return a ready-to-use image augmentation pipeline.

    Three preset strengths covering common use cases:
        - "light":  Safe augmentations, minimal distortion.
                    Good for: medical imaging, satellite imagery.
        - "medium": Balanced. Good for: most classification tasks.
        - "heavy":  Aggressive. Good for: small datasets, high variance.

    Args:
        strength: "light" | "medium" | "heavy"

    Returns:
        AugmentationPipeline instance.

    Example:
        pipeline = build_image_pipeline("medium")
        from PIL import Image
        img = Image.open("dog.jpg")
        aug_img = pipeline(img)
    """
    if strength == "light":
        return AugmentationPipeline([
            random_flip,
            lambda img: random_brightness(img, 0.9, 1.1),
        ], p=0.8)
    elif strength == "medium":
        return AugmentationPipeline([
            random_flip,
            lambda img: random_rotate(img, 15),
            lambda img: random_brightness(img, 0.75, 1.25),
            lambda img: random_contrast(img, 0.80, 1.20),
            lambda img: add_gaussian_noise(img, std=0.03),
        ], p=0.9)
    elif strength == "heavy":
        return AugmentationPipeline([
            random_flip,
            lambda img: random_rotate(img, 30),
            lambda img: random_crop(img, 0.75),
            lambda img: random_brightness(img, 0.6, 1.4),
            lambda img: add_gaussian_noise(img, std=0.06),
            lambda img: cutout(img, n_holes=2, hole_size=0.2),
        ], p=1.0)
    else:
        raise ValueError(f"Unknown strength: {strength}. Use 'light', 'medium', or 'heavy'.")

#  AUDIO AUGMENTATION# Requires: pip install librosa soundfile numpy

def pitch_shift(audio: np.ndarray, sr: int, semitones: float = 2.0) -> np.ndarray:
    """
    Shift the pitch of an audio signal by a number of semitones.

    A positive semitones value = higher pitch.
    A negative value = lower pitch.
    ±2–4 semitones is realistic for speech; ±12 = one octave.

    Args:
        audio:    1D numpy array (mono waveform, float32).
        sr:       Sample rate in Hz (e.g., 22050).
        semitones: How many semitones to shift.

    Returns:
        Pitch-shifted audio array.

    Example:
        import librosa
        y, sr = librosa.load("speech.wav")
        y_high = pitch_shift(y, sr, semitones=3)
    """
    try:
        import librosa
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    except ImportError:
        logger.warning("librosa not installed. Run: pip install librosa")
        return audio


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """
    Speed up or slow down audio without changing pitch.

    Args:
        audio: 1D audio array.
        rate:  Speed multiplier. 1.0 = original, 1.2 = 20% faster,
               0.8 = 20% slower. Keep in [0.75, 1.25] for realism.

    Example:
        y_fast = time_stretch(y, rate=1.15)
    """
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)
    except ImportError:
        logger.warning("librosa not installed.")
        return audio


def add_background_noise(
    audio: np.ndarray,
    noise_factor: float = 0.01,
    noise_type: str = "white",
) -> np.ndarray:
    """
    Mix audio with background noise.

    Args:
        audio:        Clean audio array.
        noise_factor: Volume of noise relative to signal (0.005–0.05 typical).
        noise_type:   "white" | "pink" | "brown"
                      - white: flat spectrum (hiss)
                      - pink:  1/f spectrum (more natural sounding)
                      - brown: 1/f² spectrum (rumble)

    Returns:
        Noisy audio array.

    Example:
        y_noisy = add_background_noise(y, noise_factor=0.02, noise_type="pink")
    """
    n = len(audio)
    if noise_type == "white":
        noise = np.random.randn(n)
    elif noise_type == "pink":
        # Approximate pink noise via filtered white noise
        f = np.fft.rfftfreq(n)
        f[0] = 1e-6  # avoid division by zero
        power = 1.0 / np.sqrt(f)
        noise = np.fft.irfft(np.fft.rfft(np.random.randn(n)) * power, n=n)
    elif noise_type == "brown":
        f = np.fft.rfftfreq(n)
        f[0] = 1e-6
        power = 1.0 / f
        noise = np.fft.irfft(np.fft.rfft(np.random.randn(n)) * power, n=n)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    noise = noise / (np.max(np.abs(noise)) + 1e-9)  # normalize
    return audio + noise_factor * noise


def time_mask(audio: np.ndarray, max_mask_fraction: float = 0.1) -> np.ndarray:
    """
    Zero-out a random time segment (SpecAugment / TimeMask).

    Teaches models to not rely on any single time window.
    Used in speech recognition (SpecAugment, 2019) — very effective.

    Args:
        audio:             Audio array.
        max_mask_fraction: Maximum fraction of signal to silence.

    Example:
        y_masked = time_mask(y, max_mask_fraction=0.15)
    """
    audio = audio.copy()
    n = len(audio)
    mask_len = random.randint(1, int(n * max_mask_fraction))
    start = random.randint(0, n - mask_len)
    audio[start : start + mask_len] = 0.0
    return audio


def build_audio_pipeline(strength: str = "medium") -> AugmentationPipeline:
    """
    Return a preset audio augmentation pipeline.

    Args:
        strength: "light" | "medium" | "heavy"

    Returns:
        AugmentationPipeline.

    Usage:
        import librosa
        pipeline = build_audio_pipeline("medium")
        # pipeline expects (audio, sr) tuple
        y, sr = librosa.load("audio.wav")
        y_aug, sr_aug = pipeline((y, sr))
    """
    def _pipeline(audio_sr_tuple, _strength):
        y, sr = audio_sr_tuple
        if _strength == "light":
            if random.random() > 0.5:
                y = add_background_noise(y, noise_factor=0.005)
        elif _strength == "medium":
            if random.random() > 0.5:
                y = pitch_shift(y, sr, semitones=random.uniform(-1.5, 1.5))
            if random.random() > 0.5:
                y = add_background_noise(y, noise_factor=0.015)
            if random.random() > 0.5:
                y = time_mask(y, max_mask_fraction=0.1)
        elif _strength == "heavy":
            y = pitch_shift(y, sr, semitones=random.uniform(-3, 3))
            y = time_stretch(y, rate=random.uniform(0.85, 1.15))
            y = add_background_noise(y, noise_factor=random.uniform(0.01, 0.04),
                                     noise_type=random.choice(["white", "pink"]))
            y = time_mask(y, max_mask_fraction=0.15)
        return y

    return AugmentationPipeline([lambda x: _pipeline(x, strength)])

#  TABULAR AUGMENTATION — Class imbalance handling
def gaussian_feature_noise(
    X: "pd.DataFrame",
    noise_std_fraction: float = 0.02,
    columns: Optional[list[str]] = None,
) -> "pd.DataFrame":
    """
    Add small Gaussian noise to numeric columns.

    A simple tabular augmentation that generates slightly perturbed
    copies of existing rows — useful when you have very few samples.

    Args:
        X:                  Input DataFrame.
        noise_std_fraction: Noise = fraction of each column's std dev.
        columns:            Columns to perturb. Default = all numeric.

    Returns:
        Perturbed copy of X.

    Example:
        X_aug = gaussian_feature_noise(X, noise_std_fraction=0.05)
    """
    import pandas as pd
    X = X.copy()
    num_cols = columns or X.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        std = X[col].std()
        X[col] += np.random.normal(0, noise_std_fraction * std, len(X))
    return X


def smote_like_oversample(
    X: "pd.DataFrame",
    y: "pd.Series",
    target_class: any = None,
    n_synthetic: int = 100,
    k_neighbors: int = 5,
) -> tuple["pd.DataFrame", "pd.Series"]:
    """
    A simplified SMOTE-like synthetic minority oversampling.

    SMOTE (Synthetic Minority Over-sampling TEchnique) creates synthetic
    samples by interpolating between a minority-class sample and one of
    its k nearest neighbors.

    Reference: Chawla et al. (2002) — JAIR.

    Args:
        X:             Feature DataFrame (numeric columns only).
        y:             Target Series.
        target_class:  The minority class to oversample. Auto-detected if None.
        n_synthetic:   Number of synthetic samples to generate.
        k_neighbors:   Neighbors used for interpolation.

    Returns:
        (X_resampled, y_resampled) with synthetic samples appended.

    Note: For production, use imbalanced-learn:
        pip install imbalanced-learn
        from imblearn.over_sampling import SMOTE

    Example:
        X_res, y_res = smote_like_oversample(X, y, n_synthetic=200)
        print(y_res.value_counts())
    """
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    if target_class is None:
        target_class = y.value_counts().idxmin()

    minority_X = X[y == target_class].select_dtypes(include=np.number).values

    if len(minority_X) < k_neighbors + 1:
        logger.warning("Too few minority samples for SMOTE. Returning original data.")
        return X, y

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(minority_X)
    _, indices = nn.kneighbors(minority_X)

    synthetic_rows = []
    for _ in range(n_synthetic):
        idx = random.randint(0, len(minority_X) - 1)
        neighbor_idx = random.choice(indices[idx][1:])  # skip self
        alpha = random.random()
        new_sample = minority_X[idx] + alpha * (minority_X[neighbor_idx] - minority_X[idx])
        synthetic_rows.append(new_sample)

    syn_X = pd.DataFrame(synthetic_rows,
                         columns=X.select_dtypes(include=np.number).columns)
    # Fill non-numeric columns with mode
    for col in X.select_dtypes(exclude=np.number).columns:
        syn_X[col] = X[col].mode()[0]
    syn_y = pd.Series([target_class] * n_synthetic, name=y.name)

    X_res = pd.concat([X, syn_X[X.columns]], ignore_index=True)
    y_res = pd.concat([y, syn_y], ignore_index=True)
    logger.info(f"SMOTE: {len(X)} → {len(X_res)} samples "
                f"({n_synthetic} synthetic '{target_class}' added)")
    return X_res, y_res

#  QUICK-START DEMO
if __name__ == "__main__":
    print("=" * 60)
    print("  augmentation.py — Quick Demo")
    print("=" * 60)

    # ── Text ──────────────────────────────────────────────────
    print("\n TEXT AUGMENTATION")
    sample = "The movie was absolutely fantastic and beautifully directed."
    print(f"  Original:          {sample}")
    print(f"  Synonym swap:      {synonym_replacement(sample, p=0.3)}")
    print(f"  Random deletion:   {random_deletion(sample, p=0.2)}")
    print(f"  Random swap:       {random_swap(sample, n=2)}")
    print(f"  Keyboard noise:    {keyboard_noise(sample, p=0.05)}")

    texts  = [sample, "Terrible waste of time and money."]
    labels = [1, 0]
    aug_texts, aug_labels = augment_text_batch(texts, labels, n_augmented=2)
    print(f"\n  Augmented batch: {len(texts)} → {len(aug_texts)} samples")

    # ── Image (PIL required) ──────────────────────────────────
    try:
        from PIL import Image
        print("\n  IMAGE AUGMENTATION")
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        pipeline = build_image_pipeline("medium")
        aug_img = pipeline(img)
        print(f"  Original size: {img.size}")
        print(f"  Augmented size: {aug_img.size}  (same — pipeline preserves shape)")
        print("  Applied: flip, rotate, brightness, contrast, noise")
    except ImportError:
        print("\n  IMAGE: Install Pillow → pip install pillow")

    # ── Tabular ───────────────────────────────────────────────
    try:
        import pandas as pd
        print("\n TABULAR AUGMENTATION")
        X = pd.DataFrame({
            "age":    [25, 30, 35, 40, 22],
            "salary": [40000, 55000, 70000, 85000, 38000],
        })
        y = pd.Series([0, 1, 1, 1, 0], name="target")
        X_noisy = gaussian_feature_noise(X, noise_std_fraction=0.05)
        print(f"  Original:\n{X.head(2)}")
        print(f"  With noise:\n{X_noisy.head(2)}")
    except ImportError:
        print("\n TABULAR: Install pandas → pip install pandas")

    print("\n Augmentation demo complete.")
"""
╔══════════════════════════════════════════════════════════════════════╗
║              image_preprocessing.py — Image Pipeline                 ║
║                                                                      ║
║  Complete preprocessing pipeline for computer vision tasks.          ║
║                                                                      ║
║  Pipeline steps (typical order):                                     ║
║    1. Load      — from file, URL, bytes, or numpy array              ║
║    2. Validate  — check format, size, mode                           ║
║    3. Resize    — to target dimensions                               ║
║    4. Convert   — mode changes (RGB↔Grayscale, RGBA stripping)       ║
║    5. Enhance   — denoise, sharpen, equalize histogram               ║
║    6. Normalize — pixel values to [0,1] or z-score                   ║
║    7. Export    — numpy array, tensor, or saved file                 ║
║                                                                      ║
║  Dependencies:                                                       ║
║    pip install pillow numpy                                          ║
║    pip install opencv-python         # optional, for advanced ops    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import logging
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("image_preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Type alias for clarity throughout this module
ImageLike = Union[Image.Image, np.ndarray, str, Path, bytes]

#   Loading and basic format conversion
def load_image(source: ImageLike) -> Image.Image:
    """
    Load an image from any common source into a PIL Image.

    Handles:
      - File path (str or Path)
      - Bytes / BytesIO
      - numpy array (H×W or H×W×C)
       
      - Existing PIL Image (passthrough)

    Args:
        source: Any image-like object.

    Returns:
        PIL Image in its original color mode.

    Example:
        img = load_image("photo.jpg")
        img = load_image(Path("images/dog.png"))
        img = load_image(np.zeros((224, 224, 3), dtype=np.uint8))
    """
    if isinstance(source, Image.Image):
        return source
    elif isinstance(source, (str, Path)):
        return Image.open(str(source))
    elif isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        return Image.open(source)
    elif isinstance(source, np.ndarray):
        return numpy_to_pil(source)
    else:
        raise TypeError(f"Cannot load image from type: {type(source)}")


def to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB (3-channel color).

    Handles all common modes:
      - RGBA: drops alpha channel (composites on white background)
      - L (grayscale): replicates the single channel to all 3
      - P (palette): expands to RGB
      - CMYK: converts color space

    Most deep learning models expect 3-channel RGB input.

    Example:
        img = load_image("photo_with_transparency.png")  # RGBA
        img = to_rgb(img)  # now RGB, safe for CNNs
    """
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        # Composite RGBA onto a white background
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])  # 3 = alpha channel
        return bg
    return image.convert("RGB")


def to_grayscale(image: Image.Image, keep_channels: int = 1) -> Image.Image:
    """
    Convert image to grayscale.

    Args:
        image:         Input PIL Image.
        keep_channels: 1 = true grayscale (L), 3 = grayscale as 3-channel RGB
                       (needed if model expects 3 channels).

    Example:
        gray = to_grayscale(img)           # single channel
        gray3 = to_grayscale(img, 3)       # 3 identical channels
    """
    gray = image.convert("L")
    if keep_channels == 3:
        return gray.convert("RGB")
    return gray


def pil_to_numpy(image: Image.Image, dtype: type = np.uint8) -> np.ndarray:
    """
    Convert a PIL Image to a numpy array.

    Output shape:
      - RGB:       (H, W, 3)
      - Grayscale: (H, W)    (not (H, W, 1))
      - RGBA:      (H, W, 4)

    Args:
        image: PIL Image.
        dtype: numpy dtype. np.uint8 = [0, 255], np.float32 = [0.0, 255.0]

    Example:
        arr = pil_to_numpy(img)              # uint8 [0, 255]
        arr = pil_to_numpy(img, np.float32)  # float32 [0.0, 255.0]
    """
    return np.array(image, dtype=dtype)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a numpy array back to a PIL Image.

    Handles:
      - float arrays [0.0, 1.0] → scales to [0, 255]
      - float arrays [0.0, 255.0] → no scaling needed
      - uint8 arrays → used directly

    Example:
        arr = np.zeros((256, 256, 3), dtype=np.float32)
        img = numpy_to_pil(arr)
    """
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


def image_info(image: Image.Image) -> dict:
    """
    Return metadata about an image: size, mode, aspect ratio, channels.

    Useful for validation and debugging before processing.

    Example:
        info = image_info(img)
        print(info)
        # {'width': 1920, 'height': 1080, 'mode': 'RGB',
        #  'channels': 3, 'aspect_ratio': 1.78, 'pixels': 2073600}
    """
    w, h = image.size
    mode_channels = {"RGB": 3, "RGBA": 4, "L": 1, "P": 1, "CMYK": 4, "LA": 2}
    return {
        "width":        w,
        "height":       h,
        "mode":         image.mode,
        "channels":     mode_channels.get(image.mode, "?"),
        "aspect_ratio": round(w / h, 3),
        "pixels":       w * h,
        "format":       getattr(image, "format", None),
    }

#  Resize strategies
def resize(
    image: Image.Image,
    size: Union[int, tuple[int, int]],
    strategy: str = "stretch",
    resampling: int = Image.LANCZOS,
) -> Image.Image:
    """
    Resize an image using one of several strategies.

    Strategies (choose based on your use case):

        "stretch":     Resize to exact (w, h). May distort aspect ratio.
                       Use when: model requires exact size and distortion OK.

        "fit":         Resize to fit WITHIN size, preserving aspect ratio.
                       No cropping. May result in smaller image.
                       Use when: aspect ratio must be preserved.

        "fill":        Resize and crop to fill EXACTLY size, centered.
                       Aspect ratio preserved; some content cropped.
                       Use when: you need a fixed grid (e.g., CNN batches).

        "pad":         Fit within size and pad with grey/black to fill.
                       No cropping, no distortion. May add borders.
                       Use when: content at all corners must be visible.

    Args:
        image:      PIL Image.
        size:       Target size. int → square (size, size), tuple → (W, H).
        strategy:   "stretch" | "fit" | "fill" | "pad"
        resampling: Interpolation filter.
                    LANCZOS = best quality (for downscaling).
                    BILINEAR = faster.
                    NEAREST = no interpolation (for masks/labels!).

    Returns:
        Resized PIL Image.

    Example:
        # Prepare for ResNet-50 (expects 224×224 RGB)
        img = resize(img, 224, strategy="fill")
        img = to_rgb(img)
    """
    if isinstance(size, int):
        size = (size, size)
    target_w, target_h = size
    orig_w, orig_h = image.size

    if strategy == "stretch":
        return image.resize((target_w, target_h), resampling)

    elif strategy == "fit":
        img = image.copy()
        img.thumbnail((target_w, target_h), resampling)
        return img

    elif strategy == "fill":
        # Scale to fill, then center-crop
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = image.resize((new_w, new_h), resampling)
        left  = (new_w - target_w) // 2
        upper = (new_h - target_h) // 2
        return img.crop((left, upper, left + target_w, upper + target_h))

    elif strategy == "pad":
        image.thumbnail((target_w, target_h), resampling)
        padded = Image.new(image.mode, (target_w, target_h),
                           color=(128,) * len(image.getbands()))
        offset_x = (target_w - image.width) // 2
        offset_y = (target_h - image.height) // 2
        padded.paste(image, (offset_x, offset_y))
        return padded

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Use 'stretch', 'fit', 'fill', or 'pad'.")

#  Enhancement and noise reduction
def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness. factor < 1 = darker, > 1 = brighter. 1.0 = original."""
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast. factor < 1 = low contrast, > 1 = high. 1.0 = original."""
    return ImageEnhance.Contrast(image).enhance(factor)


def adjust_sharpness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust sharpness. 0.0 = blurred, 1.0 = original, 2.0 = extra sharp."""
    return ImageEnhance.Sharpness(image).enhance(factor)


def equalize_histogram(image: Image.Image) -> Image.Image:
    """
    Apply histogram equalization to improve contrast.

    Redistributes pixel intensity values to span the full [0, 255] range.
    Effective for images that appear washed out or very dark.

    Note: Works best on grayscale. For color images, converts to grayscale
    first. For per-channel equalization, use clahe_equalize().

    Example:
        img_eq = equalize_histogram(dark_image)
    """
    if image.mode != "L":
        image = image.convert("L")
    return ImageOps.equalize(image)


def denoise(image: Image.Image, radius: int = 2) -> Image.Image:
    """
    Apply median filter for noise reduction.

    Median filtering replaces each pixel with the median of its neighbors.
    Effective for salt-and-pepper noise while preserving edges better
    than a Gaussian blur.

    Args:
        image:  PIL Image.
        radius: Filter kernel radius. 2 = 5×5 kernel (aggressive).
                1 = 3×3 (gentle). Larger = more smoothing.

    Example:
        clean = denoise(noisy_image, radius=1)
    """
    size = 2 * radius + 1
    return image.filter(ImageFilter.MedianFilter(size=size))


def auto_level(image: Image.Image) -> Image.Image:
    """
    Auto-level: stretch image to use full [0, 255] range.

    Simple but effective contrast enhancement that rescales the darkest
    pixel to 0 and the brightest to 255.

    Example:
        leveled = auto_level(washed_out_image)
    """
    return ImageOps.autocontrast(image)

#   Normalization for neural networks
def normalize_pixels(
    arr: np.ndarray,
    method: str = "minmax",
    mean: Optional[Union[float, list[float]]] = None,
    std: Optional[Union[float, list[float]]] = None,
) -> np.ndarray:
    """
    Normalize pixel values for model input.

    Methods:

        "minmax":   Scale to [0.0, 1.0] by dividing by 255.
                    Most common. Used by most modern frameworks.
                    arr / 255.0

        "zscore":   Standardize using channel mean and std.
                    Centered around 0, spread ≈ 1.
                    Used by ImageNet-pretrained models.
                    (arr/255 - mean) / std

        "imagenet": Apply ImageNet mean/std (common for transfer learning).
                    mean=[0.485, 0.456, 0.406]
                    std =[0.229, 0.224, 0.225]

        "tanh":     Scale to [-1.0, 1.0].
                    Used by GANs and some architectures.
                    arr / 127.5 - 1.0

    Args:
        arr:    numpy array, shape (H,W,C) or (H,W), dtype uint8 [0,255]
                or float32 [0.0, 1.0].
        method: Normalization method (see above).
        mean:   Per-channel mean for "zscore". float or [R_mean, G_mean, B_mean].
        std:    Per-channel std for "zscore".

    Returns:
        Normalized float32 array.

    Example:
        arr = pil_to_numpy(img)
        arr_norm = normalize_pixels(arr, method="imagenet")
        # Ready for a ResNet pretrained on ImageNet
    """
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0   # bring to [0, 1] first

    if method == "minmax":
        return arr

    elif method == "zscore":
        if mean is None or std is None:
            mean = arr.mean(axis=(0, 1)) if arr.ndim == 3 else arr.mean()
            std  = arr.std(axis=(0, 1))  if arr.ndim == 3 else arr.std()
        mean = np.array(mean, dtype=np.float32)
        std  = np.array(std,  dtype=np.float32)
        return (arr - mean) / (std + 1e-8)

    elif method == "imagenet":
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return (arr - imagenet_mean) / imagenet_std

    elif method == "tanh":
        return arr * 2.0 - 1.0

    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Use 'minmax', 'zscore', 'imagenet', or 'tanh'.")


def denormalize_pixels(
    arr: np.ndarray,
    method: str = "minmax",
    mean: Optional[Union[float, list[float]]] = None,
    std: Optional[Union[float, list[float]]] = None,
) -> np.ndarray:
    """
    Reverse normalization — convert model output back to viewable [0, 255].

    Useful for visualizing what a network sees after normalization.

    Args: Same as normalize_pixels — use the same method and params.

    Example:
        # Undo ImageNet normalization to display the image
        arr_display = denormalize_pixels(arr_normalized, method="imagenet")
        img = numpy_to_pil(arr_display)
    """
    arr = arr.astype(np.float32)

    if method == "minmax":
        pass  # already [0, 1] → just clip and multiply below
    elif method == "zscore":
        if mean is None or std is None:
            raise ValueError("mean and std required to denormalize zscore.")
        arr = arr * np.array(std) + np.array(mean)
    elif method == "imagenet":
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std  = np.array([0.229, 0.224, 0.225])
        arr = arr * imagenet_std + imagenet_mean
    elif method == "tanh":
        arr = (arr + 1.0) / 2.0

    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

#  PIPELINE — All steps in one composable object
class ImagePreprocessingPipeline:
    """
    Configurable, composable image preprocessing pipeline.

    Applies steps in this order (if enabled):
      1. Load → PIL Image
      2. Convert color mode
      3. Resize
      4. Enhance (brightness, contrast, sharpness)
      5. Denoise
      6. Convert to numpy
      7. Normalize pixel values

    Example ( resize for a CNN):
        pipeline = ImagePreprocessingPipeline(target_size=224)
        arr = pipeline.transform("cat.jpg")
        # arr.shape = (224, 224, 3), dtype=float32, values in [0,1]

    Example (intermediate — ImageNet transfer learning):
        pipeline = ImagePreprocessingPipeline(
            target_size=224,
            color_mode="rgb",
            resize_strategy="fill",
            normalize_method="imagenet",
        )
        arr = pipeline.transform(pil_image)

    Example (advanced — medical grayscale):
        pipeline = ImagePreprocessingPipeline(
            target_size=(512, 512),
            color_mode="grayscale",
            resize_strategy="pad",
            equalize=True,
            denoise_radius=1,
            normalize_method="zscore",
        )
        arr = pipeline.transform(xray_path)
    """

    def __init__(
        self,
        # Sizing
        target_size: Optional[Union[int, tuple[int, int]]] = 224,
        resize_strategy: str = "fill",
        # Color
        color_mode: str = "rgb",    # "rgb" | "grayscale" | "keep"
        # Enhancement
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        sharpness: Optional[float] = None,
        equalize: bool = False,
        denoise_radius: Optional[int] = None,
        auto_level_: bool = False,
        # Normalization
        normalize_method: Optional[str] = "minmax",
        normalize_mean: Optional[list[float]] = None,
        normalize_std: Optional[list[float]] = None,
        # Output
        output_channels_first: bool = False,  # False: (H,W,C), True: (C,H,W) for PyTorch
    ):
        self.target_size          = target_size
        self.resize_strategy      = resize_strategy
        self.color_mode           = color_mode
        self.brightness           = brightness
        self.contrast             = contrast
        self.sharpness            = sharpness
        self.equalize             = equalize
        self.denoise_radius       = denoise_radius
        self.auto_level_          = auto_level_
        self.normalize_method     = normalize_method
        self.normalize_mean       = normalize_mean
        self.normalize_std        = normalize_std
        self.output_channels_first = output_channels_first

    def transform(self, source: ImageLike) -> np.ndarray:
        """
        Apply full pipeline to one image.

        Returns:
            numpy float32 array of shape (H, W, C) or (C, H, W).
        """
        # Step 1: Load
        img = load_image(source)

        # Step 2: Color conversion
        if self.color_mode == "rgb":
            img = to_rgb(img)
        elif self.color_mode == "grayscale":
            img = to_grayscale(img)

        # Step 3: Resize
        if self.target_size is not None:
            img = resize(img, self.target_size, strategy=self.resize_strategy)

        # Step 4: Enhancement
        if self.brightness is not None:
            img = adjust_brightness(img, self.brightness)
        if self.contrast is not None:
            img = adjust_contrast(img, self.contrast)
        if self.sharpness is not None:
            img = adjust_sharpness(img, self.sharpness)
        if self.equalize:
            img = equalize_histogram(img)
        if self.denoise_radius is not None:
            img = denoise(img, radius=self.denoise_radius)
        if self.auto_level_:
            img = auto_level(img)

        # Step 5: Convert to numpy
        arr = pil_to_numpy(img, np.float32)

        # Step 6: Normalize
        if self.normalize_method:
            arr = normalize_pixels(
                arr,
                method=self.normalize_method,
                mean=self.normalize_mean,
                std=self.normalize_std,
            )

        # Step 7: Channel format
        if self.output_channels_first and arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) → (C,H,W)

        return arr

    def transform_batch(
        self,
        sources: list[ImageLike],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Process a batch of images and stack them into a single array.

        Returns:
            numpy array of shape (N, H, W, C) or (N, C, H, W).

        Example:
            paths = list(Path("images/").glob("*.jpg"))
            batch = pipeline.transform_batch(paths)
            # batch.shape = (N, 224, 224, 3)
        """
        arrays = []
        for i, source in enumerate(sources):
            try:
                arrays.append(self.transform(source))
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                continue
            if show_progress and i > 0 and i % 100 == 0:
                logger.info(f"Processed {i}/{len(sources)} images…")

        if not arrays:
            raise ValueError("No images were successfully processed.")
        result = np.stack(arrays, axis=0)
        logger.info(f"Batch complete: {result.shape}, dtype={result.dtype}")
        return result

#   UTILITIES — Save, convert, validate
def save_image(
    image: Union[Image.Image, np.ndarray],
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save an image to disk.

    Args:
        image:   PIL Image or numpy array.
        path:    Output file path. Extension determines format.
                 .jpg / .jpeg → JPEG, .png → PNG, .webp → WebP, etc.
        quality: JPEG/WebP quality (1–100). Only used for lossy formats.

    Example:
        save_image(processed_img, "output/result.jpg", quality=90)
        save_image(arr, "output/result.png")
    """
    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lower()
    if fmt in (".jpg", ".jpeg", ".webp"):
        image.save(str(path), quality=quality, optimize=True)
    else:
        image.save(str(path))
    logger.debug(f"Saved: {path}")


def batch_convert(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_format: str = "jpg",
    target_size: Optional[int] = None,
    recursive: bool = False,
) -> int:
    """
    Batch-convert all images in a folder to a target format/size.

    Args:
        input_dir:     Source directory.
        output_dir:    Destination directory (created if missing).
        target_format: Output format extension: "jpg", "png", "webp".
        target_size:   If set, resize (largest side) to this many pixels.
        recursive:     Also process images in subdirectories.

    Returns:
        Number of images converted.

    Example:
        n = batch_convert("raw_images/", "processed/", target_format="webp", target_size=512)
        print(f"Converted {n} images")
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}
    pattern = "**/*" if recursive else "*"
    paths = [p for p in input_dir.glob(pattern) if p.suffix.lower() in valid_exts]

    count = 0
    for path in paths:
        try:
            img = load_image(path)
            img = to_rgb(img)
            if target_size:
                img.thumbnail((target_size, target_size), Image.LANCZOS)
            rel = path.relative_to(input_dir).with_suffix(f".{target_format}")
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(img, out_path)
            count += 1
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")

    logger.info(f"Batch convert complete: {count}/{len(paths)} images")
    return count

# QUICK-START DEMO
if __name__ == "__main__":
    print("=" * 60)
    print("  image_preprocessing.py — Demo (synthetic images)")
    print("=" * 60)

    # Create a test image (gradient)
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    g = 100
    h, w = 480, 640
    y = np.linspace(0, 1, h).reshape(h, 1)

    r = (y * 255).astype(np.uint8)
    g = np.full((h, 1), 100, dtype=np.uint8)
    b = ((1 - y) * 255).astype(np.uint8)

    row = np.concatenate([r, g, b], axis=1)
    arr = np.repeat(row[:, None, :], w, axis=1)

    
    test_img = numpy_to_pil(arr)
    print(f"\n Created test image: {image_info(test_img)}")

    # resize
    resized = resize(test_img, 224, strategy="fill")
    print(f"   Resized (fill): {image_info(resized)}")

    # Intermediate: enhance
    bright = adjust_brightness(resized, 1.3)
    sharp  = adjust_sharpness(bright, 1.5)
    print(f"   Enhanced: brightness +30%, sharpness +50%")

    # Advanced: full pipeline
    print("\n Full pipeline (ImageNet normalization):")
    pipeline = ImagePreprocessingPipeline(
        target_size=224,
        color_mode="rgb",
        resize_strategy="fill",
        brightness=1.1,
        normalize_method="imagenet",
        output_channels_first=False,
    )
    arr_out = pipeline.transform(test_img)
    print(f"   Output shape: {arr_out.shape}")
    print(f"   dtype: {arr_out.dtype}")
    print(f"   Value range: [{arr_out.min():.3f}, {arr_out.max():.3f}]")
    print(f"   (ImageNet normalization centers values around ~0)")

    # PyTorch-style channels-first
    pipeline_pt = ImagePreprocessingPipeline(target_size=224, output_channels_first=True)
    arr_pt = pipeline_pt.transform(test_img)
    print(f"\n   PyTorch format (C,H,W): {arr_pt.shape}")

    # Batch
    images = [test_img, test_img, test_img]
    batch  = pipeline.transform_batch(images, show_progress=False)
    print(f"\n   Batch output: {batch.shape}")

    print("\n Image preprocessing demo complete.")
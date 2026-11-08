"""Upload validation shared by the web app and the model API (Task 4.2). No TensorFlow needed."""
import io
import os
import re
import unicodedata

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "MPO"}
MAX_PIXELS = 40_000_000  # decompression-bomb guard


class UploadError(ValueError):
    """User-facing validation error. .code is the HTTP status to return."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


def sanitize_filename(name: str) -> str:
    """Same idea as werkzeug.secure_filename: strip paths, odd chars, keep it short."""
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name[:100] or "upload"


def validate_upload(filename: str, data: bytes, max_bytes: int) -> dict:
    """Return image info or raise UploadError. Works purely in memory - nothing touches disk."""
    safe = sanitize_filename(filename)
    ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    if not data:
        raise UploadError("No file was uploaded or the file is empty.")
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '.{ext or '?'}'. Allowed: "
                          f"{', '.join(sorted(ALLOWED_EXTENSIONS))}.", 415)
    if len(data) > max_bytes:
        raise UploadError(f"File is too large ({len(data) / 1e6:.1f} MB). Limit is "
                          f"{max_bytes / 1e6:.0f} MB.", 413)
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, w, h = im.format, im.width, im.height
            if w * h > MAX_PIXELS:
                raise UploadError("Image dimensions are too large.", 413)
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # full decode: catches truncated / fake images
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError("The file has an image extension but could not be decoded as an image.", 422)
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"Image format {fmt} is not supported.", 415)
    return {"filename": safe, "format": fmt, "width": w, "height": h, "size_bytes": len(data)}

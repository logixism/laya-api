import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from systemone_model_api.config import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_TOTAL_IMAGE_BYTES,
)
from systemone_model_api.models import ImageInput

_IMAGE_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


def decode_images(inputs: list[ImageInput]) -> list[Image.Image]:
    """Decode embedded images only; never resolve user-supplied URLs or paths."""
    images = []
    total_bytes = 0
    try:
        for item in inputs:
            if isinstance(item, str):
                header, encoded = item.split(",", 1)
                content_type = header[5:].split(";", 1)[0].lower()
            else:
                content_type, encoded = item.content_type, item.base64
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Images must contain valid base64 data") from exc
            total_bytes += len(data)
            if len(data) > MAX_IMAGE_BYTES or total_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("Images exceed the 4 MiB per-image or 8 MiB total limit")
            try:
                with Image.open(BytesIO(data)) as image:
                    if image.format != _IMAGE_FORMATS[content_type]:
                        raise ValueError("Image content does not match its content type")
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Images must not exceed 16 megapixels")
                    images.append(image.convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("Invalid or oversized image data") from exc
        return images
    except BaseException:
        for image in images:
            image.close()
        raise

import base64
import unittest
from io import BytesIO

from PIL import Image
from pydantic import TypeAdapter, ValidationError

from systemone_model_api.config import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_TOTAL_IMAGE_BYTES
from systemone_model_api.media import decode_images
from systemone_model_api.models import Base64Image, ImageInput


def png_bytes(size=(2, 2)):
    with Image.new("RGB", size, (12, 34, 56)) as image, BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def image_input(data, content_type="image/png"):
    return Base64Image(content_type=content_type, base64=base64.b64encode(data).decode("ascii"))


class EmbeddedImageTests(unittest.TestCase):
    def test_data_url_and_object_preserve_image_pixels(self):
        encoded = image_input(png_bytes())
        images = decode_images([encoded, "data:image/png;base64," + encoded.base64])
        try:
            for image in images:
                self.assertEqual(image.size, (2, 2))
                self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))
        finally:
            for image in images:
                image.close()

    def test_remote_urls_are_not_image_inputs(self):
        with self.assertRaises(ValidationError):
            TypeAdapter(ImageInput).validate_python("https://example.com/private.png")

    def test_mime_type_cannot_disguise_image_contents(self):
        with self.assertRaises(ValueError):
            decode_images([image_input(png_bytes(), "image/jpeg")])

    def test_decoded_byte_limit_handles_base64_rounding(self):
        # These lengths encode to the same base64 length; schema validation alone
        # cannot distinguish them. Trailing PNG data is otherwise decodable.
        data = png_bytes().ljust(MAX_IMAGE_BYTES, b"\0")
        images = decode_images([image_input(data)])
        for image in images:
            image.close()
        oversized = image_input(data + b"\0")
        with self.assertRaises(ValueError):
            decode_images([oversized])

    def test_individually_valid_images_cannot_exceed_total_bytes(self):
        size = MAX_TOTAL_IMAGE_BYTES // 3 + 1
        encoded = image_input(png_bytes().ljust(size, b"\0"))
        with self.assertRaises(ValueError):
            decode_images([encoded, encoded, encoded])

    def test_compressed_image_cannot_exceed_pixel_limit(self):
        data = png_bytes((1000, MAX_IMAGE_PIXELS // 1000 + 1))
        with self.assertRaises(ValueError):
            decode_images([image_input(data)])


if __name__ == "__main__":
    unittest.main()

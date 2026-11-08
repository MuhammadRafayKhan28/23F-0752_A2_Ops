import pytest
from conftest import png_bytes

from serving.validation import UploadError, sanitize_filename, validate_upload

MB = 1024 * 1024


def test_valid_png_accepted():
    info = validate_upload("cat.png", png_bytes(), MB)
    assert info["format"] == "PNG" and info["width"] == 64 and info["height"] == 48


def test_wrong_extension_rejected():
    with pytest.raises(UploadError) as e:
        validate_upload("notes.txt", b"hello", MB)
    assert e.value.code == 415


def test_fake_image_rejected():
    with pytest.raises(UploadError) as e:
        validate_upload("photo.jpg", b"\xff\xd8\xff" + b"not really a jpeg" * 50, MB)
    assert e.value.code == 422


def test_truncated_image_rejected():
    data = png_bytes((300, 300))
    with pytest.raises(UploadError):
        validate_upload("cut.png", data[: len(data) // 2], MB)


def test_oversize_rejected():
    with pytest.raises(UploadError) as e:
        validate_upload("big.png", png_bytes() + b"0" * (2 * MB), MB)
    assert e.value.code == 413


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd.png", "passwd.png"),
    ("C:\\Users\\x\\my photo (1).JPG", "my_photo_1_.JPG"),
    ("<script>.png", "script_.png"),
])
def test_sanitize(raw, expected):
    assert sanitize_filename(raw) == expected

import io
from PIL import Image
from frontend.utils import validate_and_convert_image


def make_test_image(format='PNG'):
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    b = io.BytesIO()
    img.save(b, format=format)
    b.seek(0)
    # give a filename attribute to simulate upload
    b.name = f'test_image.{format.lower()}'
    return b


def test_validate_convert_png():
    f = make_test_image('PNG')
    is_valid, img, bytes_io, err = validate_and_convert_image(f)
    assert is_valid is True
    assert img is not None
    assert bytes_io is not None
    assert err is None
    # bytes_io should contain JPEG header
    header = bytes_io.getvalue()[:4]
    assert header[:2] == b'\xff\xd8' or header.startswith(b'JFIF')


def test_validate_unsupported_extension():
    # Test unsupported file extension
    f = io.BytesIO(b'fake image data')
    f.name = 'test.xyz'  # unsupported extension
    is_valid, img, bytes_io, err = validate_and_convert_image(f)
    assert is_valid is False
    assert img is None
    assert bytes_io is None
    assert 'Unsupported file extension: .xyz' in err


def test_validate_empty_file():
    # Test empty file
    f = io.BytesIO(b'')
    f.name = 'empty.jpg'
    is_valid, img, bytes_io, err = validate_and_convert_image(f)
    assert is_valid is False
    assert img is None 
    assert bytes_io is None
    assert 'Empty file' in err

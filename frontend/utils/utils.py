import io
from PIL import Image
import requests

# รองรับ HEIC format
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "heic", "heif", "webp", "bmp", "tiff"]
    FORMAT_DISPLAY = "JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF"
except ImportError:
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    FORMAT_DISPLAY = "JPG, JPEG, PNG, WebP, BMP, TIFF"

MAX_FILE_SIZE_MB = 10


def validate_and_convert_image(uploaded_file):
    """Validate uploaded image, enforce size and extension limits, convert to RGB JPEG bytes.

    Returns: (is_valid, pil_image, jpeg_bytesio, error_message)
    """
    try:
        # Read raw bytes from the uploaded file-like object
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        if hasattr(uploaded_file, 'read'):
            file_bytes = uploaded_file.read()
        else:
            file_bytes = getattr(uploaded_file, 'getvalue', lambda: b'')()

        if not file_bytes:
            return False, None, None, 'Empty file or unreadable upload'

        # Check size
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, None, None, f'File too large (> {MAX_FILE_SIZE_MB} MB)'

        # Check extension when available
        filename = getattr(uploaded_file, 'name', '') or ''
        if filename:
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext not in SUPPORTED_FORMATS:
                return False, None, None, f'Unsupported file extension: .{ext}'

        # Open with PIL from bytes and convert
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to JPEG bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        return True, img, img_byte_arr, None
    except Exception as e:
        return False, None, None, str(e)


def send_predict_request(files, api_url, timeout=60):
    """Send POST to /predict with given files dict.

    files: dict where values are tuples (filename, fileobj, mime)
    Returns: requests.Response
    """
    url = api_url.rstrip('/') + '/predict'
    # requests expects file-like objects for file tuples; ensure pointer at start
    prepared = {}
    for k, v in files.items():
        fname, fileobj, mime = v
        try:
            fileobj.seek(0)
        except Exception:
            pass
        prepared[k] = (fname, fileobj, mime)

    resp = requests.post(url, files=prepared, timeout=timeout)
    return resp

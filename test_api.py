import requests
from PIL import Image
import numpy as np
from io import BytesIO

# Create a simple test image
print("Creating test image...")
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)

# Convert to bytes
img_buffer = BytesIO()
img.save(img_buffer, format='JPEG')
img_bytes = img_buffer.getvalue()

# Make API request
print("Making API request...")
files = {
    'front': ('test.jpg', img_bytes, 'image/jpeg'),
    'back': ('test.jpg', img_bytes, 'image/jpeg')
}

try:
    response = requests.post('http://127.0.0.1:8001/predict', files=files, timeout=30)
    print(f'Status Code: {response.status_code}')

    if response.status_code == 200:
        result = response.json()
        print('SUCCESS: API returned prediction!')
        print(f"Top prediction: {result.get('top1', {}).get('class_name', 'Unknown')}")
        print(f"Confidence: {result.get('top1', {}).get('confidence', 0):.4f}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"AI Mode: {result.get('ai_mode', 'unknown')}")
    else:
        print(f'ERROR: {response.text}')

except Exception as e:
    print(f'Request failed: {str(e)}')
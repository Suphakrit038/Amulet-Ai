"""
Utilities for the Streamlit frontend.

This module provides a robust validate_and_convert_image function used by
frontend.main_app to validate uploads from file_uploader or camera_input and
convert them into a high-quality JPEG BytesIO for API submission.

Keeping this here ensures main_app.py can import it cleanly without relying on
fallback code paths. The implementation mirrors the advanced checks in the app
and is self-contained (no Streamlit imports).
"""

from __future__ import annotations

import io
from typing import Tuple, Optional

import numpy as np

try:
	# Pillow
	from PIL import Image, ImageFilter, ImageStat
except Exception as e:  # pragma: no cover - runtime import guard
	raise

# Optional HEIF/HEIC support
try:  # pragma: no cover - optional dependency
	from pillow_heif import register_heif_opener

	register_heif_opener()
	SUPPORTED_FORMATS = [
		"jpg",
		"jpeg",
		"png",
		"heic",
		"heif",
		"webp",
		"bmp",
		"tiff",
	]
except Exception:  # pragma: no cover - fallback when pillow-heif not present
	SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]

# Optional OpenCV checks
try:  # pragma: no cover - optional dependency
	import cv2  # type: ignore

	OPENCV_AVAILABLE = True
except Exception:  # pragma: no cover
	cv2 = None  # type: ignore
	OPENCV_AVAILABLE = False


MAX_FILE_SIZE_MB = 10


def _get_file_bytes(uploaded_file) -> bytes:
	"""Read all bytes from a Streamlit UploadedFile-like object safely."""
	try:
		# Reset pointer if supported
		if hasattr(uploaded_file, "seek"):
			uploaded_file.seek(0)
	except Exception:
		pass

	if hasattr(uploaded_file, "read"):
		return uploaded_file.read()

	# Fallback for objects exposing getvalue
	getter = getattr(uploaded_file, "getvalue", None)
	return getter() if callable(getter) else b""


def validate_and_convert_image(uploaded_file) -> Tuple[bool, Optional[Image.Image], Optional[io.BytesIO], str]:
	"""Advanced image quality validation and conversion.

	Returns:
		(is_valid, pil_image_or_none, jpeg_bytesio_or_none, message)
	"""
	try:
		file_bytes = _get_file_bytes(uploaded_file)

		if not file_bytes:
			return False, None, None, "❌ ไฟล์ว่างเปล่าหรือไม่สามารถอ่านได้"

		if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
			return False, None, None, f"❌ ไฟล์ใหญ่เกินไป (> {MAX_FILE_SIZE_MB} MB)"

		filename = getattr(uploaded_file, "name", "") or ""
		if filename:
			ext = filename.rsplit(".", 1)[-1].lower()
			if ext not in SUPPORTED_FORMATS:
				return False, None, None, f"❌ ไฟล์ประเภทนี้ไม่รองรับ: .{ext}"

		# Open image via Pillow
		img = Image.open(io.BytesIO(file_bytes))

		# 1) Minimum resolution
		width, height = img.size
		min_dimension = 200
		if width < min_dimension or height < min_dimension:
			return (
				False,
				None,
				None,
				f"❌ ภาพมีความละเอียดต่ำเกินไป (ต้องมีขนาดอย่างน้อย {min_dimension}x{min_dimension} พิกเซล)",
			)

		# Normalize to RGB
		if img.mode != "RGB":
			img = img.convert("RGB")

		# 2) Basic brightness and contrast using PIL
		stat = ImageStat.Stat(img)
		mean_brightness = sum(stat.mean) / max(1, len(stat.mean))
		if mean_brightness < 30:
			return False, None, None, "❌ ภาพมืดเกินไป กรุณาถ่ายในที่ที่มีแสงเพียงพอ"
		if mean_brightness > 240:
			return False, None, None, "❌ ภาพสว่างเกินไป อาจมีแสงแฟลชส่องจนเกิน"

		# 3) Unusual aspect ratio
		aspect_ratio = max(width, height) / min(width, height)
		if aspect_ratio > 3:
			return False, None, None, "❌ ภาพมีอัตราส่วนที่ผิดปกติ กรุณาถ่ายภาพพระเครื่องในมุมที่เหมาะสม"

		# 4) Blur detection via PIL blur-difference heuristic
		gray_img = img.convert("L")
		blur_img = gray_img.filter(ImageFilter.BLUR)

		gray_array = np.array(gray_img)
		blur_array = np.array(blur_img)
		diff = float(np.mean(np.abs(gray_array.astype(float) - blur_array.astype(float))))
		if diff < 5:
			return False, None, None, "❌ ภาพเบลอหรือไม่คมชัด กรุณาถ่ายภาพใหม่ให้คมชัดขึ้น"

		# 5) Color diversity
		img_array = np.array(img)
		color_std = float(np.std(img_array))
		if color_std < 20:
			return False, None, None, "❌ ภาพมีรายละเอียดน้อยเกินไป อาจเป็นภาพสีเดียวหรือภาพเบลอ"

		# Advanced checks using OpenCV if available
		if OPENCV_AVAILABLE and cv2 is not None:
			img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

			# Laplacian variance for sharpness
			laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
			blur_threshold = 100.0
			if laplacian_var < blur_threshold:
				return (
					False,
					None,
					None,
					f"❌ ภาพเบลอหรือไม่คมชัด (คะแนนความคมชัด: {laplacian_var:.1f} ต้องมีมากกว่า {blur_threshold:.0f})",
				)

			# Edge density
			edges = cv2.Canny(gray, 50, 150)
			edge_density = float(np.sum(edges > 0)) / float(edges.shape[0] * edges.shape[1])
			if edge_density < 0.05:
				return False, None, None, "❌ ภาพไม่มีรายละเอียดเพียงพอ อาจเกิดจากการเขย่าตัวหรือการเคลื่อนไหว"

			# Detect significant contours
			contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			significant_contours = [c for c in contours if cv2.contourArea(c) > (width * height * 0.01)]
			if len(significant_contours) < 1:
				return False, None, None, "❌ ไม่พบวัตถุที่มีรูปร่างชัดเจนในภาพ กรุณาถ่ายภาพพระเครื่องให้ชัดเจน"

			# Saturation guard
			hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
			mean_saturation = float(np.mean(hsv[:, :, 1]))
			if mean_saturation > 200:
				return False, None, None, "❌ ภาพมีสีสันจัดจ้านเกินไป อาจไม่ใช่ภาพพระเครื่องจริง"

			# Noise level
			noise_level = float(cv2.meanStdDev(gray)[1][0][0])
			if noise_level > 50:
				return False, None, None, f"❌ ภาพมีสัญญาณรบกวนมากเกินไป (ระดับสัญญาณรบกวน: {noise_level:.1f})"

			quality_score = min(100.0, (laplacian_var / blur_threshold) * 50.0 + (edge_density * 1000.0))
		else:
			# Fallback quality score without OpenCV
			quality_score = min(100.0, diff * 5.0 + (color_std / 3.0))

		# Convert to high-quality JPEG
		img_byte_arr = io.BytesIO()
		img.save(img_byte_arr, format="JPEG", quality=95)
		img_byte_arr.seek(0)

		success_msg = f"✅ ภาพผ่านการตรวจสอบคุณภาพ (คะแนน: {quality_score:.1f}/100)"
		return True, img, img_byte_arr, success_msg

	except Exception as e:  # pragma: no cover - runtime safety
		return False, None, None, f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}"


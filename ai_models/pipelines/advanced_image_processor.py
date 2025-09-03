"""
🔬 Advanced Image Processor for Maximum Quality
ระบบประมวลผลรูปภาพขั้นสูงสำหรับความชัดสูงสุด
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class AdvancedImageProcessor:
    """ระบบประมวลผลรูปภาพขั้นสูงเพื่อความชัดสูงสุด"""
    
    def __init__(self):
        """Initialize advanced image processor"""
        self.target_size = (512, 512)  # เพิ่มเป็น 512x512 สำหรับความชัดสูง
        self.preserve_aspect_ratio = True
        self.enhancement_settings = {
            'sharpness': 1.8,      # เพิ่มความคมชัดสูง
            'contrast': 1.3,       # เพิ่ม contrast
            'brightness': 1.1,     # ปรับความสว่างเล็กน้อย
            'color': 1.2           # เพิ่มความอิ่มสี
        }
        
    def process_for_maximum_clarity(self, image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        ประมวลผลรูปภาพเพื่อความชัดสูงสุด - ไม่มี augmentation
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (processed_image, numpy_array)
        """
        try:
            # Step 1: Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info("🎨 Converted image to RGB mode")
            
            # Step 2: Super-resolution upscaling (if image is small)
            original_size = image.size
            if min(original_size) < 512:
                image = self._upscale_image(image)
                logger.info(f"📏 Upscaled from {original_size} to {image.size}")
            
            # Step 3: Noise reduction
            image = self._reduce_noise(image)
            
            # Step 4: Advanced sharpening
            image = self._advanced_sharpen(image)
            
            # Step 5: Contrast and brightness optimization
            image = self._optimize_contrast_brightness(image)
            
            # Step 6: Edge enhancement
            image = self._enhance_edges(image)
            
            # Step 7: Resize to target size (preserving aspect ratio)
            processed_image = self._smart_resize(image, self.target_size)
            
            # Step 8: Convert to high-quality numpy array
            img_array = self._to_high_quality_array(processed_image)
            
            logger.info("✅ Maximum clarity processing completed")
            return processed_image, img_array
            
        except Exception as e:
            logger.error(f"❌ Image processing failed: {e}")
            # Fallback to basic processing
            return self._fallback_processing(image)
    
    def _upscale_image(self, image: Image.Image) -> Image.Image:
        """Super-resolution upscaling using LANCZOS"""
        width, height = image.size
        
        # Calculate new size (minimum 512 on short side)
        if width < height:
            new_width = 512
            new_height = int(height * 512 / width)
        else:
            new_height = 512
            new_width = int(width * 512 / height)
        
        # Use best resampling algorithm
        upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return upscaled
    
    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Advanced noise reduction"""
        # Convert to OpenCV format for advanced processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            cv_image, 
            None, 
            h=10,           # Filter strength
            hColor=10,      # Filter strength for color
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Convert back to PIL
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return Image.fromarray(denoised_rgb)
    
    def _advanced_sharpen(self, image: Image.Image) -> Image.Image:
        """Advanced sharpening using unsharp mask"""
        # Convert to numpy for advanced processing
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Create Gaussian blur for unsharp mask
        blurred = cv2.GaussianBlur(img_array, (0, 0), 1.0)
        
        # Unsharp mask
        sharpened = img_array + 0.8 * (img_array - blurred)  # Strong sharpening
        sharpened = np.clip(sharpened, 0, 1)
        
        # Convert back to PIL
        sharpened_uint8 = (sharpened * 255).astype(np.uint8)
        return Image.fromarray(sharpened_uint8)
    
    def _optimize_contrast_brightness(self, image: Image.Image) -> Image.Image:
        """Optimize contrast and brightness using histogram equalization"""
        # Convert to OpenCV for CLAHE (Contrast Limited Adaptive Histogram Equalization)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(enhanced_rgb)
    
    def _enhance_edges(self, image: Image.Image) -> Image.Image:
        """Enhance edges without creating artifacts"""
        # Use PIL's built-in edge enhancement
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(self.enhancement_settings['sharpness'])
        
        # Additional contrast enhancement
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(self.enhancement_settings['contrast'])
        
        return enhanced
    
    def _smart_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Smart resize that preserves aspect ratio and maximizes quality"""
        if not self.preserve_aspect_ratio:
            return image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Calculate dimensions to preserve aspect ratio
        orig_width, orig_height = image.size
        target_width, target_height = target_size
        
        ratio = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)
        
        # Resize with best quality
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create target size canvas with white background
        final_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Center the image
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        final_image.paste(resized, (x, y))
        
        return final_image
    
    def _to_high_quality_array(self, image: Image.Image) -> np.ndarray:
        """Convert to high-quality numpy array"""
        # Convert to float32 for maximum precision
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension if needed
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _fallback_processing(self, image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """Fallback processing if advanced methods fail"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic resize
        processed = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Basic enhancement
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(1.5)
        
        img_array = np.array(processed, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return processed, img_array
    
    def create_high_quality_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """สร้าง thumbnail คุณภาพสูง - ไม่ใช่ 150x150 แล้ว"""
        # ใช้ขนาดใหญ่กว่าเพื่อความชัด
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # ปรับปรุงคุณภาพก่อน resize
        enhanced = self._enhance_edges(image)
        
        # Resize ด้วยคุณภาพสูงสุด
        thumbnail = enhanced.resize(size, Image.Resampling.LANCZOS)
        
        return thumbnail

    def get_image_quality_metrics(self, image: Image.Image) -> Dict:
        """วิเคราะห์คุณภาพของรูปภาพ"""
        img_array = np.array(image)
        
        # Calculate various quality metrics
        sharpness = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        
        # Contrast measurement
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = gray.std()
        
        # Brightness
        brightness = np.mean(img_array)
        
        # Signal-to-noise ratio estimation
        snr = np.mean(img_array) / (np.std(img_array) + 1e-10)
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'snr': float(snr),
            'resolution': image.size
        }

# Global instance
advanced_processor = AdvancedImageProcessor()

def process_image_max_quality(image: Image.Image) -> Tuple[Image.Image, np.ndarray, Dict]:
    """
    Main function to process image for maximum quality
    
    Returns:
        Tuple of (processed_image, numpy_array, quality_metrics)
    """
    processed_img, img_array = advanced_processor.process_for_maximum_clarity(image)
    quality_metrics = advanced_processor.get_image_quality_metrics(processed_img)
    
    return processed_img, img_array, quality_metrics

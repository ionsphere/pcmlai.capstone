import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import random
from enum import Enum


class DegradationType(Enum):
    FADING = "fading"
    STAIN = "stain"
    TEAR = "tear"
    PILLING = "pilling"
    WRINKLE = "wrinkle"
    DISCOLORATION = "discoloration"
    FRAYING = "fraying"
    WEAR_PATTERN = "wear_pattern"


class ConditionLevel(Enum):
    PRISTINE = 10        # Brand new, perfect condition
    EXCELLENT = 9        # Like new, minimal wear
    VERY_GOOD = 8        # Slight wear, great condition
    GOOD = 7             # Noticeable but minor wear
    FAIR_PLUS = 6        # Moderate wear, still good
    FAIR = 5             # Average wear, acceptable
    FAIR_MINUS = 4       # Significant wear
    POOR = 3             # Heavy wear, damaged
    VERY_POOR = 2        # Severe damage
    UNWEARABLE = 1       # Completely worn out


class SyntheticDegradationPipeline:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def apply_degradation(
        self,
        image: np.ndarray,
        condition_level: int,
        degradation_types: Optional[List[DegradationType]] = None,
        intensity: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:
        if condition_level < 1 or condition_level > 10:
            raise ValueError("Condition level must be between 1 and 10")
        
        img = Image.fromarray(image)
        metadata = {
            'condition_level': condition_level,
            'applied_effects': [],
            'intensity': intensity
        }
        
        severity = (10 - condition_level) / 10.0 * intensity
        
        if degradation_types is None:
            degradation_types = self._select_random_degradations(condition_level)
        
        for deg_type in degradation_types:
            if deg_type == DegradationType.FADING:
                img = self._apply_fading(img, severity)
                metadata['applied_effects'].append('fading')
            
            elif deg_type == DegradationType.STAIN:
                img = self._apply_stains(img, severity)
                metadata['applied_effects'].append('stain')
            
            elif deg_type == DegradationType.TEAR:
                img = self._apply_tears(img, severity)
                metadata['applied_effects'].append('tear')
            
            elif deg_type == DegradationType.PILLING:
                img = self._apply_pilling(img, severity)
                metadata['applied_effects'].append('pilling')
            
            elif deg_type == DegradationType.WRINKLE:
                img = self._apply_wrinkles(img, severity)
                metadata['applied_effects'].append('wrinkle')
            
            elif deg_type == DegradationType.DISCOLORATION:
                img = self._apply_discoloration(img, severity)
                metadata['applied_effects'].append('discoloration')
            
            elif deg_type == DegradationType.FRAYING:
                img = self._apply_fraying(img, severity)
                metadata['applied_effects'].append('fraying')
            
            elif deg_type == DegradationType.WEAR_PATTERN:
                img = self._apply_wear_patterns(img, severity)
                metadata['applied_effects'].append('wear_pattern')
        
        result = np.array(img)
        
        return result, metadata
    
    def _select_random_degradations(self, condition_level: int) -> List[DegradationType]:
        all_types = list(DegradationType)
        
        if condition_level >= 8:  # Excellent condition
            num_effects = random.randint(1, 2)
        elif condition_level >= 6:  # Good condition
            num_effects = random.randint(2, 3)
        elif condition_level >= 4:  # Fair condition
            num_effects = random.randint(3, 5)
        else:  # Poor condition
            num_effects = random.randint(4, 6)
        
        return random.sample(all_types, min(num_effects, len(all_types)))
    
    def _apply_fading(self, img: Image.Image, severity: float) -> Image.Image:
        enhancer = ImageEnhance.Color(img)
        fade_factor = 1.0 - (severity * 0.5)  # Reduce saturation by up to 50%
        img = enhancer.enhance(fade_factor)
        
        enhancer = ImageEnhance.Contrast(img)
        contrast_factor = 1.0 - (severity * 0.2)
        img = enhancer.enhance(contrast_factor)
        
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = 1.0 + (severity * 0.1)
        img = enhancer.enhance(brightness_factor)
        
        return img
    
    def _apply_stains(self, img: Image.Image, severity: float) -> Image.Image:
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        num_stains = int(severity * 5) + 1
        
        for _ in range(num_stains):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            radius = int(random.uniform(10, 50) * severity)
            
            stain_type = random.choice(['brown', 'yellow', 'gray', 'red'])
            if stain_type == 'brown':
                color = np.array([101, 67, 33], dtype=np.float32)
            elif stain_type == 'yellow':
                color = np.array([255, 235, 59], dtype=np.float32)
            elif stain_type == 'gray':
                color = np.array([100, 100, 100], dtype=np.float32)
            else:  # red (wine, blood)
                color = np.array([139, 0, 0], dtype=np.float32)
            
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = np.clip(1 - (dist_from_center / radius), 0, 1)
            
            alpha = mask[..., np.newaxis] * (0.3 + severity * 0.4)
            img_array = img_array * (1 - alpha) + color * alpha
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_tears(self, img: Image.Image, severity: float) -> Image.Image:
        if severity < 0.3:
            return img
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        num_tears = int((severity - 0.3) * 3) + 1
        
        for _ in range(num_tears):
            x = random.randint(int(width * 0.2), int(width * 0.8))
            y = random.randint(int(height * 0.2), int(height * 0.8))
            
            tear_length = int(random.uniform(20, 80) * severity)
            tear_width = int(random.uniform(2, 10) * severity)
            
            angle = random.uniform(0, 2 * np.pi)
            dx = int(tear_length * np.cos(angle))
            dy = int(tear_length * np.sin(angle))
            
            cv2.line(img_array, (x, y), (x + dx, y + dy), (50, 50, 50), tear_width)
            
            for i in range(tear_width * 2):
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                cv2.line(img_array, 
                        (x + offset_x, y + offset_y), 
                        (x + dx + offset_x, y + dy + offset_y),
                        (80, 80, 80), 1)
        
        return Image.fromarray(img_array)
    
    def _apply_pilling(self, img: Image.Image, severity: float) -> Image.Image:
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        num_pills = int(severity * 100)
        
        for _ in range(num_pills):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            pill_size = random.randint(1, 3)
            
            if random.random() > 0.5:
                color_offset = 20
            else:
                color_offset = -20
            
            if 0 <= y < height and 0 <= x < width:
                base_color = img_array[y, x].astype(np.int16)
                pill_color = np.clip(base_color + color_offset, 0, 255).astype(np.uint8)
                
                cv2.circle(img_array, (x, y), pill_size, pill_color.tolist(), -1)
        
        return Image.fromarray(img_array)
    
    def _apply_wrinkles(self, img: Image.Image, severity: float) -> Image.Image:
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        num_wrinkles = int(severity * 5) + 2
        
        for _ in range(num_wrinkles):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(0, width - 1)
            y2 = random.randint(0, height - 1)
            
            wrinkle_width = int(20 * severity)
            
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps == 0:
                continue
                
            for i in range(steps):
                t = i / steps
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                
                for dy in range(-wrinkle_width, wrinkle_width):
                    for dx in range(-wrinkle_width, wrinkle_width):
                        px, py = x + dx, y + dy
                        if 0 <= px < width and 0 <= py < height:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < wrinkle_width:
                                darkness = 1.0 - (dist / wrinkle_width) * 0.15 * severity
                                img_array[py, px] *= darkness
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_discoloration(self, img: Image.Image, severity: float) -> Image.Image:
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        num_patches = int(severity * 3) + 1
        
        for _ in range(num_patches):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            radius = int(random.uniform(50, 150))
            
            shift_type = random.choice(['yellow', 'brown', 'gray'])
            if shift_type == 'yellow':
                color_shift = np.array([10, 10, -10]) * severity
            elif shift_type == 'brown':
                color_shift = np.array([5, 0, -10]) * severity
            else:
                color_shift = np.array([-5, -5, -5]) * severity
            
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = np.clip(1 - (dist_from_center / radius), 0, 1)
            
            for c in range(3):
                img_array[:, :, c] += mask * color_shift[c]
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_fraying(self, img: Image.Image, severity: float) -> Image.Image:
        if severity < 0.4:
            return img
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        num_frays = int((severity - 0.4) * 30)
        
        edges = ['top', 'bottom', 'left', 'right']
        edge = random.choice(edges)
        
        for _ in range(num_frays):
            if edge == 'top':
                x = random.randint(0, width - 1)
                y = random.randint(0, int(height * 0.1))
            elif edge == 'bottom':
                x = random.randint(0, width - 1)
                y = random.randint(int(height * 0.9), height - 1)
            elif edge == 'left':
                x = random.randint(0, int(width * 0.1))
                y = random.randint(0, height - 1)
            else:  # right
                x = random.randint(int(width * 0.9), width - 1)
                y = random.randint(0, height - 1)
            
            length = random.randint(5, 15)
            angle = random.uniform(-np.pi/4, np.pi/4)
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))
            
            cv2.line(img_array, (x, y), (x + dx, y + dy), (100, 100, 100), 1)
        
        return Image.fromarray(img_array)
    
    def _apply_wear_patterns(self, img: Image.Image, severity: float) -> Image.Image:
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        wear_areas = [
            (width // 2, height // 3, width // 6),  # Center-top (collar)
            (width // 3, height * 2 // 3, width // 8),  # Left-lower (knee/elbow)
            (width * 2 // 3, height * 2 // 3, width // 8),  # Right-lower (knee/elbow)
        ]
        
        for x, y, radius in wear_areas:
            if random.random() < severity:
                Y, X = np.ogrid[:height, :width]
                dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
                mask = np.clip(1 - (dist_from_center / radius), 0, 1)
                
                wear_factor = 0.8 + severity * 0.2
                for c in range(3):
                    img_array[:, :, c] = img_array[:, :, c] * (1 - mask * 0.2) + 255 * mask * 0.2 * severity
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def generate_condition_dataset(
        self,
        image_path: str,
        output_dir: str,
        num_variations: int = 5,
        condition_range: Tuple[int, int] = (1, 10)
    ) -> List[Dict]:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        results = []
        
        for i in range(num_variations):
            condition = random.randint(condition_range[0], condition_range[1])
            
            degraded_img, metadata = self.apply_degradation(img, condition)
            
            output_filename = f"{base_name}_condition_{condition}_var_{i}.jpg"
            output_file = output_path / output_filename
            
            degraded_img_bgr = cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), degraded_img_bgr)
            
            metadata['source_image'] = image_path
            metadata['output_path'] = str(output_file)
            metadata['variation_id'] = i
            results.append(metadata)
        
        return results


def batch_generate_synthetic_data(
    input_dir: str,
    output_dir: str,
    num_variations: int = 5,
    condition_range: Tuple[int, int] = (1, 10),
    max_images: Optional[int] = None,
    verbose = False
) -> Dict[str, any]:
    pipeline = SyntheticDegradationPipeline()
    
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_files = [
        f for f in input_path.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    if max_images:
        image_files = image_files[:max_images]
    
    if verbose:
        print(f"Found {len(image_files)} images to process")
    
    all_metadata = []
    successful = 0
    failed = 0
    
    for idx, img_file in enumerate(image_files):
        try:
            if verbose:
                print(f"Processing {idx + 1}/{len(image_files)}: {img_file.name}")
            
            metadata_list = pipeline.generate_condition_dataset(
                str(img_file),
                output_dir,
                num_variations,
                condition_range
            )
            
            all_metadata.extend(metadata_list)
            successful += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            failed += 1
    
    stats = {
        'total_source_images': len(image_files),
        'successful': successful,
        'failed': failed,
        'total_generated': len(all_metadata),
        'variations_per_image': num_variations,
        'condition_range': condition_range,
        'metadata': all_metadata
    }
    
    return stats

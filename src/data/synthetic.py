"""
Synthetic Wear & Tear Generation Module

This module provides comprehensive tools for generating realistic clothing degradation
effects to augment training data for the clothing condition assessment model.

Author: GenSpark AI Developer
Date: 2025-01-22
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import random
from enum import Enum


class DegradationType(Enum):
    """Types of clothing degradation effects."""
    FADING = "fading"
    STAIN = "stain"
    TEAR = "tear"
    PILLING = "pilling"
    WRINKLE = "wrinkle"
    DISCOLORATION = "discoloration"
    FRAYING = "fraying"
    WEAR_PATTERN = "wear_pattern"


class ConditionLevel(Enum):
    """Clothing condition levels (1-10 scale)."""
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
    """
    Main pipeline for applying synthetic wear and tear to clothing images.
    
    This class provides methods to generate realistic degradation effects
    at various severity levels.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the degradation pipeline.
        
        Args:
            seed: Random seed for reproducibility
        """
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
        """
        Apply degradation effects to an image based on condition level.
        
        Args:
            image: Input image as numpy array (RGB)
            condition_level: Target condition level (1-10)
            degradation_types: Specific degradation types to apply (None = random selection)
            intensity: Overall intensity multiplier (0.0-1.0)
        
        Returns:
            Tuple of (degraded_image, metadata_dict)
        """
        if condition_level < 1 or condition_level > 10:
            raise ValueError("Condition level must be between 1 and 10")
        
        # Convert to PIL Image for easier manipulation
        img = Image.fromarray(image)
        metadata = {
            'condition_level': condition_level,
            'applied_effects': [],
            'intensity': intensity
        }
        
        # Determine severity based on condition level
        # Higher condition = less degradation
        severity = (10 - condition_level) / 10.0 * intensity
        
        # Select degradation types if not specified
        if degradation_types is None:
            degradation_types = self._select_random_degradations(condition_level)
        
        # Apply each degradation type
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
        
        # Convert back to numpy array
        result = np.array(img)
        
        return result, metadata
    
    def _select_random_degradations(self, condition_level: int) -> List[DegradationType]:
        """Select random degradation types based on condition level."""
        all_types = list(DegradationType)
        
        # More severe conditions get more degradation types
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
        """
        Apply color fading effect to simulate sun exposure and washing.
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Faded PIL Image
        """
        # Reduce color saturation
        enhancer = ImageEnhance.Color(img)
        fade_factor = 1.0 - (severity * 0.5)  # Reduce saturation by up to 50%
        img = enhancer.enhance(fade_factor)
        
        # Reduce contrast slightly
        enhancer = ImageEnhance.Contrast(img)
        contrast_factor = 1.0 - (severity * 0.2)
        img = enhancer.enhance(contrast_factor)
        
        # Add slight brightness increase (faded look)
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = 1.0 + (severity * 0.1)
        img = enhancer.enhance(brightness_factor)
        
        return img
    
    def _apply_stains(self, img: Image.Image, severity: float) -> Image.Image:
        """
        Apply random stain effects.
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Stained PIL Image
        """
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Number of stains based on severity
        num_stains = int(severity * 5) + 1
        
        for _ in range(num_stains):
            # Random stain position
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            # Random stain size
            radius = int(random.uniform(10, 50) * severity)
            
            # Random stain color (brownish, yellowish, grayish)
            stain_type = random.choice(['brown', 'yellow', 'gray', 'red'])
            if stain_type == 'brown':
                color = np.array([101, 67, 33], dtype=np.float32)
            elif stain_type == 'yellow':
                color = np.array([255, 235, 59], dtype=np.float32)
            elif stain_type == 'gray':
                color = np.array([100, 100, 100], dtype=np.float32)
            else:  # red (wine, blood)
                color = np.array([139, 0, 0], dtype=np.float32)
            
            # Create circular stain mask
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = np.clip(1 - (dist_from_center / radius), 0, 1)
            
            # Apply stain with alpha blending
            alpha = mask[..., np.newaxis] * (0.3 + severity * 0.4)
            img_array = img_array * (1 - alpha) + color * alpha
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_tears(self, img: Image.Image, severity: float) -> Image.Image:
        """
        Apply tear and hole effects.
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Torn PIL Image
        """
        if severity < 0.3:  # Only apply tears for moderate to severe damage
            return img
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Number of tears
        num_tears = int((severity - 0.3) * 3) + 1
        
        for _ in range(num_tears):
            # Random tear position
            x = random.randint(int(width * 0.2), int(width * 0.8))
            y = random.randint(int(height * 0.2), int(height * 0.8))
            
            # Tear size
            tear_length = int(random.uniform(20, 80) * severity)
            tear_width = int(random.uniform(2, 10) * severity)
            
            # Random tear angle
            angle = random.uniform(0, 2 * np.pi)
            dx = int(tear_length * np.cos(angle))
            dy = int(tear_length * np.sin(angle))
            
            # Draw tear as dark/shadow area
            cv2.line(img_array, (x, y), (x + dx, y + dy), (50, 50, 50), tear_width)
            
            # Add frayed edges around tear
            for i in range(tear_width * 2):
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                cv2.line(img_array, 
                        (x + offset_x, y + offset_y), 
                        (x + dx + offset_x, y + dy + offset_y),
                        (80, 80, 80), 1)
        
        return Image.fromarray(img_array)
    
    def _apply_pilling(self, img: Image.Image, severity: float) -> Image.Image:
        """
        Apply fabric pilling effect (small balls of fiber).
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Pilled PIL Image
        """
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Number of pills
        num_pills = int(severity * 100)
        
        for _ in range(num_pills):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            # Small pills (1-3 pixels)
            pill_size = random.randint(1, 3)
            
            # Pills are slightly lighter/darker than surrounding
            if random.random() > 0.5:
                color_offset = 20
            else:
                color_offset = -20
            
            # Get surrounding color and modify
            if 0 <= y < height and 0 <= x < width:
                base_color = img_array[y, x].astype(np.int16)
                pill_color = np.clip(base_color + color_offset, 0, 255).astype(np.uint8)
                
                # Draw small circle
                cv2.circle(img_array, (x, y), pill_size, pill_color.tolist(), -1)
        
        return Image.fromarray(img_array)
    
    def _apply_wrinkles(self, img: Image.Image, severity: float) -> Image.Image:
        """
        Apply wrinkle effects using displacement mapping.
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Wrinkled PIL Image
        """
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Create wrinkle displacement map
        num_wrinkles = int(severity * 5) + 2
        
        for _ in range(num_wrinkles):
            # Random wrinkle line
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(0, width - 1)
            y2 = random.randint(0, height - 1)
            
            # Create wrinkle influence mask
            wrinkle_width = int(20 * severity)
            
            # For each point on the line, darken nearby pixels
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps == 0:
                continue
                
            for i in range(steps):
                t = i / steps
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                
                # Darken in a band around the line
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
        """
        Apply uneven discoloration patches.
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Discolored PIL Image
        """
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Number of discoloration patches
        num_patches = int(severity * 3) + 1
        
        for _ in range(num_patches):
            # Random patch center
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            # Patch radius
            radius = int(random.uniform(50, 150))
            
            # Random color shift (yellow, brown, gray tint)
            shift_type = random.choice(['yellow', 'brown', 'gray'])
            if shift_type == 'yellow':
                color_shift = np.array([10, 10, -10]) * severity
            elif shift_type == 'brown':
                color_shift = np.array([5, 0, -10]) * severity
            else:
                color_shift = np.array([-5, -5, -5]) * severity
            
            # Create circular gradient mask
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = np.clip(1 - (dist_from_center / radius), 0, 1)
            
            # Apply color shift
            for c in range(3):
                img_array[:, :, c] += mask * color_shift[c]
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_fraying(self, img: Image.Image, severity: float) -> Image.Image:
        """
        Apply fraying effect at edges.
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Frayed PIL Image
        """
        if severity < 0.4:  # Only apply for moderate to severe damage
            return img
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Add frayed edges (small irregular lines at borders)
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
            
            # Draw small irregular line
            length = random.randint(5, 15)
            angle = random.uniform(-np.pi/4, np.pi/4)
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))
            
            cv2.line(img_array, (x, y), (x + dx, y + dy), (100, 100, 100), 1)
        
        return Image.fromarray(img_array)
    
    def _apply_wear_patterns(self, img: Image.Image, severity: float) -> Image.Image:
        """
        Apply general wear patterns (knee areas, elbows, collar).
        
        Args:
            img: PIL Image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Worn PIL Image
        """
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Common wear areas (proportional to image size)
        wear_areas = [
            (width // 2, height // 3, width // 6),  # Center-top (collar)
            (width // 3, height * 2 // 3, width // 8),  # Left-lower (knee/elbow)
            (width * 2 // 3, height * 2 // 3, width // 8),  # Right-lower (knee/elbow)
        ]
        
        for x, y, radius in wear_areas:
            if random.random() < severity:
                # Create wear mask
                Y, X = np.ogrid[:height, :width]
                dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
                mask = np.clip(1 - (dist_from_center / radius), 0, 1)
                
                # Lighten and desaturate worn area
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
        """
        Generate a dataset of degraded variations from a single image.
        
        Args:
            image_path: Path to source image
            output_dir: Directory to save generated images
            num_variations: Number of variations to generate
            condition_range: Range of condition levels to generate
        
        Returns:
            List of metadata dictionaries for each generated image
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Base filename
        base_name = Path(image_path).stem
        
        results = []
        
        for i in range(num_variations):
            # Random condition level within range
            condition = random.randint(condition_range[0], condition_range[1])
            
            # Apply degradation
            degraded_img, metadata = self.apply_degradation(img, condition)
            
            # Save image
            output_filename = f"{base_name}_condition_{condition}_var_{i}.jpg"
            output_file = output_path / output_filename
            
            degraded_img_bgr = cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), degraded_img_bgr)
            
            # Add to results
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
    """
    Batch generate synthetic degradation data from a directory of images.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save generated images
        num_variations: Number of variations per image
        condition_range: Range of condition levels
        max_images: Maximum number of source images to process (None = all)
    
    Returns:
        Dictionary with generation statistics and metadata
    """
    pipeline = SyntheticDegradationPipeline()
    
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Find all images
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

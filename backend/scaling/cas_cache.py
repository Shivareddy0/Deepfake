import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class PerceptualHashCache:
    def __init__(self):
        # Database structure: { dhash_hex: cached_result_dict }
        self.cache: Dict[str, Dict[str, Any]] = {}
        # Lists for vectorized Hamming distance checks
        self.hashes: List[np.ndarray] = []
        self.keys: List[str] = []

    def _compute_dhash(self, image_path: str) -> Optional[str]:
        """
        Computes a 64-bit Difference Hash (dHash) of an image.
        dHash tracks structural luminance gradients.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Video file check: read first frame
                cap = cv2.VideoCapture(image_path)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    return None
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize to 9x8 pixels (9 cols, 8 rows)
            resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
            
            # Compute difference between columns
            diff = resized[:, 1:] > resized[:, :-8] # Compare col(i) with col(i+1)
            # Flatten to 64 boolean bits
            flat_diff = diff.flatten()
            
            # Convert binary array to hex string
            hex_str = ""
            for i in range(0, 64, 4):
                nibble = flat_diff[i:i+4]
                val = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(nibble)))
                hex_str += f"{val:x}"
            return hex_str
        except Exception:
            return None

    def _hex_to_binary_array(self, hex_str: str) -> np.ndarray:
        binary_str = bin(int(hex_str, 16))[2:].zfill(64)
        return np.array([int(char) for char in binary_str], dtype=np.uint8)

    def find_near_duplicate(self, media_path: str, threshold: int = 8) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
        """
        Searches the cache database for visual near-duplicates within Hamming distance threshold.
        Returns:
            (cached_result, matching_hash, hamming_distance)
        """
        query_hash = self._compute_dhash(media_path)
        if not query_hash:
            return None, None, 64
            
        if query_hash in self.cache:
            return self.cache[query_hash], query_hash, 0
            
        if not self.hashes:
            return None, query_hash, 64
            
        # Perform vectorized Hamming distance query (ANN approximation)
        query_bits = self._hex_to_binary_array(query_hash)
        database_bits = np.array(self.hashes) # Shape: [N, 64]
        
        # Hamming distance is sum of XOR bits
        distances = np.sum(query_bits != database_bits, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist <= threshold:
            matching_hex = self.keys[min_idx]
            return self.cache[matching_hex], matching_hex, int(min_dist)
            
        return None, query_hash, int(min_dist)

    def set(self, hex_hash: str, result: Dict[str, Any]):
        """
        Caches a detection result.
        """
        if not hex_hash:
            return
        self.cache[hex_hash] = result
        binary_bits = self._hex_to_binary_array(hex_hash)
        
        # Update search index list
        if hex_hash in self.keys:
            idx = self.keys.index(hex_hash)
            self.hashes[idx] = binary_bits
        else:
            self.keys.append(hex_hash)
            self.hashes.append(binary_bits)
            
    def clear(self):
        self.cache.clear()
        self.hashes.clear()
        self.keys.clear()

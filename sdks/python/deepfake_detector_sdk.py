import os
import requests
from typing import Dict, Any, List, Optional

class DeepfakeDetectorSDK:
    def __init__(self, endpoint_url: str = "http://127.0.0.1:8000"):
        self.base_url = endpoint_url.rstrip('/')

    def detect_file(self, file_path: str, is_video: bool = False, is_synthetic: bool = False) -> Dict[str, Any]:
        """
        Uploads a media asset to run tiered deepfake detection.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found at {file_path}")
            
        url = f"{self.base_url}/detect/upload"
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            data = {
                'is_video': str(is_video).lower(),
                'is_synthetic': str(is_synthetic).lower()
            }
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()

    def submit_batch(self, media_urls: List[str], priority: str = "medium", webhook_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Submits a batch of media URLs for processing.
        """
        url = f"{self.base_url}/detect/batch"
        payload = {
            "media_urls": media_urls,
            "priority": priority,
            "webhook_url": webhook_url
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def embed_c2pa(self, file_path: str, output_path: str, creator: str = "SDK Author", edit_summary: str = "Edited via SDK") -> bool:
        """
        Uploads an image, signs and embeds C2PA content credentials, and saves the output.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found at {file_path}")
            
        url = f"{self.base_url}/c2pa/embed"
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            data = {
                'creator': creator,
                'edit_summary': edit_summary
            }
            response = requests.post(url, files=files, data=data, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as out_f:
                for chunk in response.iter_content(chunk_size=8192):
                    out_f.write(chunk)
            return True

    def verify_c2pa(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts and verifies C2PA provenance credentials and blockchain anchoring status.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found at {file_path}")
            
        url = f"{self.base_url}/c2pa/verify"
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json()

    def get_model_zoo(self) -> Dict[str, Any]:
        """
        Queries the model zoo registry performance metrics.
        """
        url = f"{self.base_url}/zoo/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

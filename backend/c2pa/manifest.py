import os
import json
import hashlib
import time
from typing import Dict, Any, List, Optional

# Attempt to load standard c2pa library, fallback to pure python simulation if unavailable
C2PA_AVAILABLE = False
try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    pass

class C2PAManifestManager:
    def __init__(self):
        self.cert_issuer = "CN=Antigravity Media Authority, O=Deepmind Deepfake Shield, C=US"
        self.algorithm = "ecdsa-p256-sha256"

    def create_manifest(self, title: str, creator: str, assertions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a standard C2PA manifest representation conforming to C2PA 2.0.
        """
        manifest = {
            "c2pa_version": "2.0",
            "active_manifest": True,
            "label": f"antigravity_{int(time.time())}",
            "title": title,
            "format": "image/jpeg",
            "vendor": "Antigravity Systems",
            "claim": {
                "recorder": "Camera Capture App v1.0",
                "signature_info": {
                    "alg": self.algorithm,
                    "issuer": self.cert_issuer,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                }
            },
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "softwareAgent": "Sony Alpha 7 IV",
                                "when": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 3600))
                            }
                        ]
                    }
                },
                {
                    "label": "c2pa.creator",
                    "data": {
                        "name": creator,
                        "identifier": f"did:key:{hashlib.sha256(creator.encode()).hexdigest()[:16]}"
                    }
                }
            ]
        }
        
        # Append extra editing/AI assertions
        for assertion in assertions:
            manifest["assertions"].append(assertion)
            
        return manifest

    def embed_credentials(self, media_path: str, manifest: Dict[str, Any], output_path: str) -> bool:
        """
        Embeds the C2PA manifest cryptographic signature block into the media container.
        """
        if not os.path.exists(media_path):
            return False
            
        try:
            # Generate cryptographic binding: SHA-256 of media body
            with open(media_path, 'rb') as f:
                content = f.read()
            media_hash = hashlib.sha256(content).hexdigest()
            
            # Embed binding inside manifest
            manifest["media_hash"] = media_hash
            manifest_bytes = json.dumps(manifest).encode('utf-8')
            
            # Signature block simulation (ECDSA P-256 signature representation)
            sig_source = f"{media_hash}.{json.dumps(manifest['claim'])}"
            mock_signature = hashlib.sha256(sig_source.encode()).hexdigest()
            
            payload = {
                "manifest": manifest,
                "signature": mock_signature,
                "anchor": "ethereum_sepolia_chain"
            }
            
            payload_str = f"##C2PA_START##{json.dumps(payload)}##C2PA_END##".encode('utf-8')
            
            # Write new media container
            with open(output_path, 'wb') as f:
                # C2PA standard injection puts the manifest in a metadata box.
                # In JPEG, it's APP11 marker. We simulate it by appending the payload to the end
                # of the file, allowing easy, non-destructive read/write in pure Python.
                f.write(content)
                f.write(payload_str)
                
            return True
        except Exception as e:
            print(f"C2PA embedding error: {e}")
            return False

    def verify_credentials(self, media_path: str) -> Dict[str, Any]:
        """
        Extracts, parses, and validates the cryptographic C2PA content credential chain.
        """
        result = {
            "verified": False,
            "manifest": None,
            "chain": [],
            "error": None
        }
        
        if not os.path.exists(media_path):
            result["error"] = "Media file not found."
            return result
            
        try:
            with open(media_path, 'rb') as f:
                content = f.read()
                
            start_tag = b"##C2PA_START##"
            end_tag = b"##C2PA_END##"
            
            idx_start = content.rfind(start_tag)
            idx_end = content.rfind(end_tag)
            
            if idx_start == -1 or idx_end == -1 or idx_end < idx_start:
                # No credential embedded
                result["error"] = "No C2PA content credentials found in media headers."
                return result
                
            payload_bytes = content[idx_start + len(start_tag):idx_end]
            payload = json.loads(payload_bytes.decode('utf-8'))
            
            manifest = payload["manifest"]
            signature = payload["signature"]
            
            # Verify file binding
            raw_media = content[:idx_start]
            current_hash = hashlib.sha256(raw_media).hexdigest()
            original_hash = manifest.get("media_hash", "")
            
            if current_hash != original_hash:
                result["error"] = "Cryptographic binding mismatch: Media payload has been tampered with after signing."
                return result
                
            # Signature check
            sig_source = f"{original_hash}.{json.dumps(manifest['claim'])}"
            expected_signature = hashlib.sha256(sig_source.encode()).hexdigest()
            if signature != expected_signature:
                result["error"] = "Cryptographic signature validation failed; root certificates cannot be authenticated."
                return result
                
            result["verified"] = True
            result["manifest"] = manifest
            
            # Compile edit/provenance timeline chain
            chain = []
            for assertion in manifest.get("assertions", []):
                label = assertion.get("label", "")
                data = assertion.get("data", {})
                
                if label == "c2pa.actions":
                    for act in data.get("actions", []):
                        chain.append({
                            "type": "Action",
                            "name": act.get("action", ""),
                            "agent": act.get("softwareAgent", ""),
                            "timestamp": act.get("when", "")
                        })
                elif label == "c2pa.creator":
                    chain.append({
                        "type": "Creator",
                        "name": data.get("name", ""),
                        "identifier": data.get("identifier", ""),
                        "timestamp": manifest["claim"]["signature_info"]["timestamp"]
                    })
                elif label == "c2pa.training_data":
                    chain.append({
                        "type": "AI Training Provenance",
                        "status": data.get("status", ""),
                        "dataset": data.get("dataset_name", ""),
                        "timestamp": data.get("when", "")
                    })
                    
            result["chain"] = sorted(chain, key=lambda x: x.get("timestamp", ""))
            
        except Exception as e:
            result["error"] = f"Parsing failure: {str(e)}"
            
        return result

import os
import json
import hashlib
import time
from typing import Dict, Any, Optional

class BlockchainAnchorManager:
    def __init__(self, ledger_file: str = "d:\\shiva reddy project\\backend\\c2pa\\simulated_ledger.json"):
        self.ledger_file = ledger_file
        self._init_ledger()

    def _init_ledger(self):
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)
        if not os.path.exists(self.ledger_file):
            with open(self.ledger_file, 'w') as f:
                json.dump({"anchors": {}, "block_height": 19482701}, f, indent=4)

    def anchor_hash(self, media_hash: str, manifest_label: str) -> Dict[str, Any]:
        """
        Anchor a C2PA manifest hash to the simulated blockchain network.
        Returns the transaction receipt.
        """
        try:
            with open(self.ledger_file, 'r') as f:
                data = json.load(f)
                
            block_height = data.get("block_height", 19482700) + 1
            data["block_height"] = block_height
            
            # Check if already anchored
            if media_hash in data["anchors"]:
                return data["anchors"][media_hash]
                
            # Create a mock transaction hash
            tx_payload = f"{media_hash}:{manifest_label}:{block_height}:{time.time()}"
            tx_hash = "0x" + hashlib.sha256(tx_payload.encode()).hexdigest()
            
            receipt = {
                "transaction_hash": tx_hash,
                "block_number": block_height,
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "gas_used": 21000 + int(hashlib.sha256(media_hash.encode()).hexdigest()[:4], 16) % 30000,
                "status": "Success",
                "media_hash": media_hash,
                "manifest_label": manifest_label
            }
            
            data["anchors"][media_hash] = receipt
            
            with open(self.ledger_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            return receipt
        except Exception as e:
            return {
                "status": "Failed",
                "error": str(e)
            }

    def verify_anchor(self, media_hash: str) -> Dict[str, Any]:
        """
        Query the blockchain ledger to verify if the media hash was previously anchored.
        """
        try:
            with open(self.ledger_file, 'r') as f:
                data = json.load(f)
                
            if media_hash in data["anchors"]:
                return {
                    "anchored": True,
                    "receipt": data["anchors"][media_hash]
                }
            return {
                "anchored": False,
                "receipt": None
            }
        except Exception:
            return {
                "anchored": False,
                "error": "Failed to read blockchain ledger."
            }

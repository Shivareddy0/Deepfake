import os
from PIL import Image
from PIL.ExifTags import TAGS
from backend.detectors.base import BaseDetector, DetectionResult

class MetadataDetector(BaseDetector):
    def __init__(self):
        super().__init__(name="MetadataForensicModule")

    def _read_xmp(self, file_path):
        """
        Scan binary image data to extract XMP XML metadata block.
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            xmp_start = data.find(b'<x:xmpmeta')
            xmp_end = data.find(b'</x:xmpmeta>')
            
            if xmp_start != -1 and xmp_end != -1:
                xmp_data = data[xmp_start:xmp_end+12].decode('utf-8', errors='ignore')
                return xmp_data
        except Exception:
            pass
        return None

    def detect(self, media_path: str, **kwargs) -> DetectionResult:
        if not os.path.exists(media_path):
            return DetectionResult(
                detector_name=self.name,
                confidence=0.0,
                explanation=f"Media file not found at {media_path}",
                evidence={"error": "File not found"}
            )
            
        evidence = {
            "exif_present": False,
            "xmp_present": False,
            "editing_software_detected": [],
            "ai_provenance_markers": [],
            "missing_camera_fingerprints": False
        }
        
        confidence = 0.0
        explanations = []

        # 1. Parse EXIF data using PIL
        try:
            with Image.open(media_path) as img:
                info = img._getexif()
                if info is not None:
                    evidence["exif_present"] = True
                    exif_data = {}
                    for tag, value in info.items():
                        decoded = TAGS.get(tag, tag)
                        exif_data[decoded] = str(value)
                        
                    # Scan for editing signatures in EXIF fields
                    software = exif_data.get("Software", "").lower()
                    if software:
                        evidence["software_value"] = software
                        for tool in ["photoshop", "gimp", "lightroom", "pixlr", "canva", "figma", "midjourney", "stable diffusion"]:
                            if tool in software:
                                evidence["editing_software_detected"].append(tool)
                                
                    # Scan for AI generation software tags
                    if "stablediffusion" in software or "midjourney" in software or "dall-e" in software:
                        evidence["ai_provenance_markers"].append(software)
                        
                    # A camera capture usually has fields like Make, Model, DateTimeOriginal, ShutterSpeedValue or FNumber.
                    # If it lacks these but has an editing Software, it is suspicious.
                    camera_fields = ["Make", "Model", "FNumber", "ExposureTime"]
                    missing_fields = [f for f in camera_fields if f not in exif_data]
                    if len(missing_fields) == len(camera_fields):
                        evidence["missing_camera_fingerprints"] = True
                else:
                    evidence["missing_camera_fingerprints"] = True
        except Exception:
            evidence["missing_camera_fingerprints"] = True

        # 2. Parse XMP data
        xmp_xml = self._read_xmp(media_path)
        if xmp_xml:
            evidence["xmp_present"] = True
            evidence["xmp_length"] = len(xmp_xml)
            
            # Check for common edit markers in XMP
            for tool in ["photoshop", "gimp", "illustrator", "creative cloud"]:
                if tool in xmp_xml.lower():
                    evidence["editing_software_detected"].append(tool)
                    
            # Check for generative AI tags (e.g., Stable Diffusion prompt logs or Adobe Firefly metadata assertions)
            if "stable-diffusion" in xmp_xml.lower() or "automatic1111" in xmp_xml.lower():
                evidence["ai_provenance_markers"].append("stable-diffusion-xmp")
            if "adobe:provenance" in xmp_xml.lower() or "firefly" in xmp_xml.lower():
                evidence["ai_provenance_markers"].append("adobe-firefly-xmp")

        # Deduplicate editing tools
        evidence["editing_software_detected"] = list(set(evidence["editing_software_detected"]))
        evidence["ai_provenance_markers"] = list(set(evidence["ai_provenance_markers"]))

        # 3. Calculate Confidence based on forensic findings
        if len(evidence["ai_provenance_markers"]) > 0:
            confidence = 0.95
            explanations.append(f"Explicit generative AI metadata assertions found: {evidence['ai_provenance_markers']}.")
        elif len(evidence["editing_software_detected"]) > 0:
            confidence = 0.65
            explanations.append(f"Image contains editing history footprints from: {evidence['editing_software_detected']}.")
            if evidence["missing_camera_fingerprints"]:
                confidence = 0.80
                explanations.append("The file also completely lacks hardware camera sensor profiles, indicating a synthetic or heavily compiled asset.")
        elif evidence["missing_camera_fingerprints"] and not evidence["exif_present"] and not evidence["xmp_present"]:
            # Standard web format, not necessarily fake, low suspicion
            confidence = 0.15
            explanations.append("Lacks camera EXIF headers and editing history tags; typical of compressed web-distributed media.")
        else:
            confidence = 0.05
            explanations.append("Intact camera sensor metadata profile detected with no traces of photo manipulation software history.")

        return DetectionResult(
            detector_name=self.name,
            confidence=confidence,
            explanation=" ".join(explanations),
            evidence=evidence
        )

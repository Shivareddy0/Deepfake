from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class DetectionResult(BaseModel):
    detector_name: str = Field(..., description="Name of the detector plugin")
    confidence: float = Field(..., description="Likelihood of media being manipulated (0.0 to 1.0)")
    explanation: str = Field(..., description="Natural language justification of the verdict")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Detailed quantitative forensic evidence metrics")
    manipulated_regions: List[Dict[str, Any]] = Field(default_factory=list, description="List of bounding boxes, spatial overlays, or temporal segments where manipulation was found")

class BaseDetector(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect(self, media_path: str, **kwargs) -> DetectionResult:
        """
        Execute detection on a media asset file path.
        """
        pass

import time
from typing import Dict, Any, List, Tuple
from backend.detectors.base import DetectionResult
from backend.fusion.bayesian_ensemble import BayesianEnsembleFusion

class TieredDetectionRouter:
    def __init__(self, detectors: Dict[str, Any], fusion_engine: BayesianEnsembleFusion):
        """
        Args:
            detectors: A dictionary mapping detector names to detector instances.
            fusion_engine: The Bayesian Meta-Classifier instance.
        """
        self.detectors = detectors
        self.fusion_engine = fusion_engine
        
        # Suspicous thresholds
        self.tier1_suspicion_threshold = 0.25
        self.tier2_suspicion_threshold = 0.50

    def process_media(self, media_path: str, is_video: bool = False, is_audio: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Executes tiered detection on the asset.
        """
        start_time = time.time()
        results: Dict[str, DetectionResult] = {}
        active_tiers_executed = []
        
        # ----------------- TIER 1 -----------------
        # Run Frequency Analyzer & Metadata forensics (Extremely fast, <10ms)
        active_tiers_executed.append("Tier 1 (Frequency & Metadata)")
        
        t1_detectors = ["FrequencyDomainAnalyzer", "MetadataForensicModule"]
        for name in t1_detectors:
            if name in self.detectors:
                results[name] = self.detectors[name].detect(media_path, **kwargs)
                
        # Calculate maximum confidence at Tier 1
        t1_max_conf = max([res.confidence for name, res in results.items()], default=0.0)
        
        # Always run Tier 2 (Spatial CNN) to ensure deep visual inspection is executed
        # using the pre-trained deep models loaded by the user.
        trigger_tier2 = True
        
        # ----------------- TIER 2 -----------------
        trigger_tier3 = False
        if trigger_tier2:
            active_tiers_executed.append("Tier 2 (Spatial CNN)")
            spatial_detector_name = "SpatialCNNDetector"
            if spatial_detector_name in self.detectors:
                results[spatial_detector_name] = self.detectors[spatial_detector_name].detect(media_path, **kwargs)
                
            # Check maximum confidence including Tier 2
            t2_max_conf = max([res.confidence for name, res in results.items()], default=0.0)
            trigger_tier3 = (t2_max_conf >= self.tier2_suspicion_threshold) or is_video or is_audio
            
        else:
            # Inject empty results for omitted detectors
            results["SpatialCNNDetector"] = DetectionResult(
                detector_name="SpatialCNNDetector",
                confidence=0.0,
                explanation="Tier 2 skipped: Asset determined authentic by Tier 1 fast forensic checks.",
                evidence={"skipped": True}
            )

        # ----------------- TIER 3 -----------------
        # Tier 3 (Temporal & Audio) is executed if video/audio is suspicious after Tier 2
        if trigger_tier3 and (is_video or is_audio):
            active_tiers_executed.append("Tier 3 (Temporal & Audio)")
            if is_video:
                # For video: run both temporal and audio detectors
                t3_detectors = ["TemporalCoherenceAnalyzer", "AudioDeepfakeDetector"]
            else:
                # For audio-only: only AudioDeepfakeDetector is meaningful
                t3_detectors = ["AudioDeepfakeDetector"]
                results["TemporalCoherenceAnalyzer"] = DetectionResult(
                    detector_name="TemporalCoherenceAnalyzer",
                    confidence=0.0,
                    explanation="Temporal analysis skipped: audio-only file has no video frames.",
                    evidence={"skipped": True}
                )
            for name in t3_detectors:
                if name in self.detectors:
                    results[name] = self.detectors[name].detect(media_path, **kwargs)
        else:
            # Inject empty results for omitted Tier 3 detectors
            skipped_msg = "Tier 3 skipped: Asset determined authentic by Tier 1/2 checks." if not trigger_tier3 else "Tier 3 skipped: Asset is a static image."
            results["TemporalCoherenceAnalyzer"] = DetectionResult(
                detector_name="TemporalCoherenceAnalyzer",
                confidence=0.0,
                explanation=skipped_msg,
                evidence={"skipped": True}
            )
            results["AudioDeepfakeDetector"] = DetectionResult(
                detector_name="AudioDeepfakeDetector",
                confidence=0.0,
                explanation=skipped_msg,
                evidence={"skipped": True}
            )

        # Compile individual detector output dict
        fused_scores = {name: res.confidence for name, res in results.items()}
        
        # For audio-only files: the AudioDeepfakeDetector is the primary signal.
        # Use it directly as the fused confidence to avoid the Bayesian ensemble
        # being drowned out by near-zero Frequency/Spatial/Temporal scores.
        if is_audio and not is_video:
            audio_conf = results.get("AudioDeepfakeDetector", None)
            if audio_conf is not None and not audio_conf.evidence.get("skipped", False):
                fused_prob = float(audio_conf.confidence)
                epistemic_unc = 0.05
                human_review = bool(0.4 < fused_prob < 0.7)
                total_latency_ms = (time.time() - start_time) * 1000.0
                return {
                    "media_path": media_path,
                    "fused_confidence": fused_prob,
                    "epistemic_uncertainty": epistemic_unc,
                    "requires_human_review": human_review,
                    "latency_ms": total_latency_ms,
                    "tiers_executed": active_tiers_executed,
                    "detector_results": {name: res.model_dump() for name, res in results.items()}
                }

        # Compute fused consensus score via Bayesian fusion
        fused_prob, epistemic_unc, human_review = self.fusion_engine.fuse_predictions(fused_scores)
        
        total_latency_ms = (time.time() - start_time) * 1000.0
        
        # Generate compiled report structure
        return {
            "media_path": media_path,
            "fused_confidence": fused_prob,
            "epistemic_uncertainty": epistemic_unc,
            "requires_human_review": human_review,
            "latency_ms": total_latency_ms,
            "tiers_executed": active_tiers_executed,
            "detector_results": {name: res.model_dump() for name, res in results.items()}
        }

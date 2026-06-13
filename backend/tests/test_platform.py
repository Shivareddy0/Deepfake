import os
import sys
import unittest
import numpy as np
import cv2
import tempfile
import json
import shutil

# Ensure backend is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.detectors.frequency import FrequencyDetector
from backend.detectors.spatial import SpatialDetector
from backend.detectors.temporal import TemporalDetector
from backend.detectors.audio import AudioDetector
from backend.detectors.metadata import MetadataDetector
from backend.c2pa.manifest import C2PAManifestManager
from backend.c2pa.blockchain import BlockchainAnchorManager
from backend.fusion.bayesian_ensemble import BayesianEnsembleFusion
from backend.scaling.tiered_router import TieredDetectionRouter
from backend.scaling.cas_cache import PerceptualHashCache

class TestDeepfakePlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create a dummy image representing a camera photo
        cls.photo_path = os.path.join(cls.temp_dir, "test_camera.jpg")
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(cls.photo_path, img)
        
        # Create a dummy video file
        cls.video_path = os.path.join(cls.temp_dir, "test_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.video_path, fourcc, 10.0, (128, 128))
        for _ in range(20):
            frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        # Create a dummy audio file (empty is fine since reading it throws exception triggering fallback simulation)
        cls.audio_path = os.path.join(cls.temp_dir, "fake_audio.wav")
        with open(cls.audio_path, "wb") as f:
            f.write(b"")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_frequency_detector(self):
        detector = FrequencyDetector()
        res = detector.detect(self.photo_path)
        self.assertEqual(res.detector_name, "FrequencyDomainAnalyzer")
        self.assertIn("radial_profile", res.evidence)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)

    def test_spatial_detector(self):
        detector = SpatialDetector()
        res = detector.detect(self.photo_path)
        self.assertEqual(res.detector_name, "SpatialCNNDetector")
        self.assertIn("gradcam_heatmap", res.evidence)
        self.assertGreater(len(res.evidence["gradcam_heatmap"]), 0)

    def test_temporal_detector(self):
        detector = TemporalDetector()
        res = detector.detect(self.video_path)
        self.assertEqual(res.detector_name, "TemporalCoherenceAnalyzer")
        self.assertIn("ear_timeline", res.evidence)
        self.assertIn("optical_flow_anomalies", res.evidence)

    def test_audio_detector(self):
        detector = AudioDetector()
        # Test on dummy path (triggering fallback)
        res = detector.detect(self.audio_path)
        self.assertEqual(res.detector_name, "AudioDeepfakeDetector")
        self.assertIn("vocal_tract_length_cm", res.evidence)
        self.assertIn("npvi_rhythm_score", res.evidence)

    def test_metadata_detector(self):
        detector = MetadataDetector()
        res = detector.detect(self.photo_path)
        self.assertEqual(res.detector_name, "MetadataForensicModule")
        self.assertFalse(res.evidence["exif_present"]) # Dummy generated image has no EXIF

    def test_c2pa_credentials(self):
        c2pa = C2PAManifestManager()
        manifest = c2pa.create_manifest("Artifact 1", "John Doe", [])
        
        output_file = os.path.join(self.temp_dir, "c2pa_output.jpg")
        success = c2pa.embed_credentials(self.photo_path, manifest, output_file)
        self.assertTrue(success)
        
        # Verify
        verify_res = c2pa.verify_credentials(output_file)
        self.assertTrue(verify_res["verified"])
        self.assertEqual(verify_res["manifest"]["title"], "Artifact 1")

    def test_blockchain_anchor(self):
        anchor = BlockchainAnchorManager(ledger_file=os.path.join(self.temp_dir, "ledger.json"))
        receipt = anchor.anchor_hash("hash123", "test_manifest")
        self.assertEqual(receipt["status"], "Success")
        self.assertEqual(receipt["media_hash"], "hash123")
        
        verify = anchor.verify_anchor("hash123")
        self.assertTrue(verify["anchored"])
        self.assertEqual(verify["receipt"]["transaction_hash"], receipt["transaction_hash"])

    def test_bayesian_fusion(self):
        fusion = BayesianEnsembleFusion()
        scores = {
            "FrequencyDomainAnalyzer": 0.9,
            "SpatialCNNDetector": 0.85,
            "TemporalCoherenceAnalyzer": 0.75,
            "AudioDeepfakeDetector": 0.1,
            "MetadataForensicModule": 0.8
        }
        prob, unc, human = fusion.fuse_predictions(scores)
        self.assertGreater(prob, 0.5)
        self.assertGreaterEqual(unc, 0.0)

    def test_tiered_router(self):
        detectors = {
            "FrequencyDomainAnalyzer": FrequencyDetector(),
            "SpatialCNNDetector": SpatialDetector(),
            "TemporalCoherenceAnalyzer": TemporalDetector(),
            "AudioDeepfakeDetector": AudioDetector(),
            "MetadataForensicModule": MetadataDetector()
        }
        fusion = BayesianEnsembleFusion()
        router = TieredDetectionRouter(detectors, fusion)
        
        # Test image (only T1 runs because it's authentic and low score)
        report = router.process_media(self.photo_path, is_video=False)
        self.assertIn("fused_confidence", report)
        self.assertIn("Tier 1 (Frequency & Metadata)", report["tiers_executed"])

    def test_perceptual_cache(self):
        cache = PerceptualHashCache()
        dummy_res = {"verdict": "fake"}
        
        # Search empty
        res, h, dist = cache.find_near_duplicate(self.photo_path)
        self.assertNil = self.assertIsNone(res)
        
        # Add to cache
        cache.set(h, dummy_res)
        
        # Search again
        res, h2, dist2 = cache.find_near_duplicate(self.photo_path)
        self.assertEqual(res["verdict"], "fake")
        self.assertEqual(dist2, 0)

if __name__ == "__main__":
    unittest.main()

import numpy as np
from typing import Dict, List, Tuple

class BayesianEnsembleFusion:
    def __init__(self):
        # We model the weights of the 5 detectors: 
        # [Frequency, Spatial, Temporal, Audio, Metadata]
        # Initially, we set reasonable mean priors reflecting detector reliability.
        self.mu_w = np.array([1.2, 1.8, 1.5, 1.4, 0.8])
        self.sigma_w = np.array([0.15, 0.2, 0.18, 0.15, 0.25])  # Standard deviations representing parameter uncertainty
        self.bias_mu = -1.2
        self.bias_sigma = 0.3
        self.temperature = 1.25 # Temperature scaling calibration factor

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def fuse_predictions(self, detector_scores: Dict[str, float]) -> Tuple[float, float, bool]:
        """
        Fuses predictions using a Bayesian Meta-Classifier with parameter sampling.
        Returns:
            - fused_probability: The mean probability of manipulation
            - epistemic_uncertainty: The variance in probability estimates (high indicates disagreement)
            - requires_human_review: Boolean flag indicating if human analyst review is recommended
        """
        # Align scores in the order of detectors:
        # [FrequencyDomainAnalyzer, SpatialCNNDetector, TemporalCoherenceAnalyzer, AudioDeepfakeDetector, MetadataForensicModule]
        score_vector = np.array([
            detector_scores.get("FrequencyDomainAnalyzer", 0.0),
            detector_scores.get("SpatialCNNDetector", 0.0),
            detector_scores.get("TemporalCoherenceAnalyzer", 0.0),
            detector_scores.get("AudioDeepfakeDetector", 0.0),
            detector_scores.get("MetadataForensicModule", 0.0)
        ])
        
        # Monte Carlo sampling from the posterior distribution of model parameters
        num_samples = 100
        sampled_probabilities = []
        
        # We sample weight vectors from the Gaussian posterior N(mu_w, diag(sigma_w^2))
        for _ in range(num_samples):
            w_sampled = np.random.normal(self.mu_w, self.sigma_w)
            b_sampled = np.random.normal(self.bias_mu, self.bias_sigma)
            
            # Logit calculation
            logit = np.dot(w_sampled, score_vector) + b_sampled
            # Apply Temperature Scaling Calibration
            calibrated_logit = logit / self.temperature
            prob = self._sigmoid(calibrated_logit)
            sampled_probabilities.append(prob)
            
        sampled_probabilities = np.array(sampled_probabilities)
        
        fused_probability = float(np.mean(sampled_probabilities))
        # Epistemic uncertainty is the variance of predictions due to parameter variability
        epistemic_uncertainty = float(np.var(sampled_probabilities))
        
        # Detect ensemble disagreement: if standard deviation of individual detector outputs
        # is very high, or epistemic uncertainty is above threshold, require human review.
        # We look at active detectors (confidence > 0.1) and check variance
        active_scores = [s for s in score_vector if s > 0.1]
        score_spread = np.var(active_scores) if len(active_scores) > 1 else 0.0
        
        # Thresholds:
        # Fused uncertainty > 0.03 or detector score spread > 0.08 suggests high disagreement
        requires_human_review = epistemic_uncertainty > 0.025 or score_spread > 0.06 or (0.4 < fused_probability < 0.65)
        
        return fused_probability, epistemic_uncertainty, bool(requires_human_review)

    def update_priors(self, feedback_data: List[Tuple[Dict[str, float], float]]):
        """
        Online adaptation/training of the meta-classifier weights using a simple gradient step.
        feedback_data is a list of tuples: (detector_scores_dict, true_label) where true_label is 1.0 (fake) or 0.0 (real).
        """
        lr = 0.05
        for scores, y_true in feedback_data:
            score_vector = np.array([
                scores.get("FrequencyDomainAnalyzer", 0.0),
                scores.get("SpatialCNNDetector", 0.0),
                scores.get("TemporalCoherenceAnalyzer", 0.0),
                scores.get("AudioDeepfakeDetector", 0.0),
                scores.get("MetadataForensicModule", 0.0)
            ])
            
            # Forward pass
            logit = np.dot(self.mu_w, score_vector) + self.bias_mu
            y_pred = self._sigmoid(logit / self.temperature)
            
            # Gradient of Binary Cross Entropy w.r.t logits
            d_logit = y_pred - y_true
            
            # Update means
            self.mu_w -= lr * d_logit * score_vector
            self.bias_mu -= lr * d_logit
            
            # Shrink standard deviations as we observe more data (reducing uncertainty)
            self.sigma_w = np.maximum(0.05, self.sigma_w * 0.98)
            self.bias_sigma = max(0.05, self.bias_sigma * 0.98)

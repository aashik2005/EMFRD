"""
Modality Contribution Analyzer for EMFRD

Analyzes and visualizes contributions from different modalities:
- Semantic (text features)
- Behavioral (graph features)
- Adversarial (robustness features)
- Metadata (review metadata)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class ModalityContributionAnalyzer:
    """
    Analyze contribution of each modality to final prediction

    Uses ablation and gate values to determine modality importance.

    Args:
        fusion_model: The multimodal fusion model
        device: Computation device
    """

    def __init__(
        self,
        fusion_model,
        device: str = "cpu",
    ):
        self.fusion_model = fusion_model
        self.device = device
        self.fusion_model.eval()

    def analyze_contributions(
        self,
        semantic_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adversarial_features: Optional[torch.Tensor] = None,
        metadata_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Analyze modality contributions for a single prediction

        Returns:
            Dictionary containing:
                - predictions: Predictions with different modality combinations
                - gates: Gate values from fusion model
                - contributions: Computed contribution scores
                - rankings: Modalities ranked by importance
        """
        with torch.no_grad():
            results = {}

            # Full prediction with all modalities
            full_output = self._predict_with_modalities(
                semantic_features,
                graph_features,
                adversarial_features,
                metadata_features,
            )
            results["full"] = full_output

            # Ablation: Remove one modality at a time
            ablations = {}

            if semantic_features is not None:
                ablations["no_semantic"] = self._predict_with_modalities(
                    None,
                    graph_features,
                    adversarial_features,
                    metadata_features,
                )

            if graph_features is not None:
                ablations["no_graph"] = self._predict_with_modalities(
                    semantic_features,
                    None,
                    adversarial_features,
                    metadata_features,
                )

            if adversarial_features is not None:
                ablations["no_adversarial"] = self._predict_with_modalities(
                    semantic_features,
                    graph_features,
                    None,
                    metadata_features,
                )

            if metadata_features is not None:
                ablations["no_metadata"] = self._predict_with_modalities(
                    semantic_features,
                    graph_features,
                    adversarial_features,
                    None,
                )

            results["ablations"] = ablations

            # Only use single modality
            single_modality = {}

            if semantic_features is not None:
                single_modality["semantic_only"] = self._predict_with_modalities(
                    semantic_features, None, None, None
                )

            if graph_features is not None:
                single_modality["graph_only"] = self._predict_with_modalities(
                    None, graph_features, None, None
                )

            if adversarial_features is not None:
                single_modality["adversarial_only"] = self._predict_with_modalities(
                    None, None, adversarial_features, None
                )

            if metadata_features is not None:
                single_modality["metadata_only"] = self._predict_with_modalities(
                    None, None, None, metadata_features
                )

            results["single_modality"] = single_modality

            # Compute contribution scores
            contributions = self._compute_contributions(results)

            # Get gate values
            gates = full_output.get("gates", {})
            gate_values = {
                key: float(value[0, 0].item()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in gates.items()
            }

            # Rank modalities
            rankings = self._rank_modalities(contributions, gate_values)

            return {
                "predictions": results,
                "contributions": contributions,
                "gates": gate_values,
                "rankings": rankings,
                "summary": self._generate_summary(contributions, gate_values, rankings),
            }

    def _predict_with_modalities(
        self,
        semantic_features: Optional[torch.Tensor],
        graph_features: Optional[torch.Tensor],
        adversarial_features: Optional[torch.Tensor],
        metadata_features: Optional[torch.Tensor],
    ) -> Dict[str, any]:
        """Make prediction with given modality combination"""
        try:
            # Check if at least one modality is present
            if all(f is None for f in [semantic_features, graph_features, adversarial_features, metadata_features]):
                return {
                    "prediction": 0,
                    "probability": 0.5,
                    "logits": torch.tensor([[0.0, 0.0]]),
                }

            outputs = self.fusion_model(
                semantic_features=semantic_features,
                graph_features=graph_features,
                adversarial_features=adversarial_features,
                metadata_features=metadata_features,
            )

            logits = outputs["logits"]
            probas = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probas, dim=-1).item()
            probability = probas[0, prediction].item()

            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "fake_probability": float(probas[0, 1].item()),
                "logits": logits,
                "gates": outputs.get("gates", {}),
            }

        except Exception as e:
            # Fallback if combination not supported
            return {
                "prediction": 0,
                "probability": 0.5,
                "fake_probability": 0.5,
                "error": str(e),
            }

    def _compute_contributions(self, results: Dict) -> Dict[str, float]:
        """
        Compute contribution score for each modality

        Contribution = (Full - Without_Modality) / Full
        """
        contributions = {}

        full_prob = results["full"]["fake_probability"]

        ablations = results.get("ablations", {})

        if "no_semantic" in ablations:
            no_semantic_prob = ablations["no_semantic"]["fake_probability"]
            contributions["semantic"] = abs(full_prob - no_semantic_prob)

        if "no_graph" in ablations:
            no_graph_prob = ablations["no_graph"]["fake_probability"]
            contributions["graph"] = abs(full_prob - no_graph_prob)

        if "no_adversarial" in ablations:
            no_adversarial_prob = ablations["no_adversarial"]["fake_probability"]
            contributions["adversarial"] = abs(full_prob - no_adversarial_prob)

        if "no_metadata" in ablations:
            no_metadata_prob = ablations["no_metadata"]["fake_probability"]
            contributions["metadata"] = abs(full_prob - no_metadata_prob)

        # Normalize to sum to 1
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _rank_modalities(
        self,
        contributions: Dict[str, float],
        gate_values: Dict[str, float],
    ) -> List[Dict[str, any]]:
        """Rank modalities by importance"""
        rankings = []

        for modality in contributions.keys():
            contribution = contributions.get(modality, 0.0)
            gate = gate_values.get(modality, 0.0)

            # Combined score: average of contribution and gate
            combined_score = (contribution + gate) / 2

            rankings.append({
                "modality": modality,
                "contribution": contribution,
                "gate_value": gate,
                "combined_score": combined_score,
            })

        # Sort by combined score
        rankings.sort(key=lambda x: x["combined_score"], reverse=True)

        return rankings

    def _generate_summary(
        self,
        contributions: Dict[str, float],
        gate_values: Dict[str, float],
        rankings: List[Dict],
    ) -> str:
        """Generate human-readable summary"""
        if not rankings:
            return "No modality analysis available."

        summary = "Modality Contribution Analysis:\n\n"

        for i, rank in enumerate(rankings, 1):
            modality = rank["modality"]
            contribution = rank["contribution"]
            gate = rank["gate_value"]

            summary += f"{i}. {modality.upper()}:\n"
            summary += f"   - Contribution: {contribution:.1%}\n"
            summary += f"   - Gate value: {gate:.1%}\n"
            summary += f"   - Combined score: {rank['combined_score']:.1%}\n\n"

        # Add interpretation
        top_modality = rankings[0]["modality"]
        summary += f"Primary decision factor: {top_modality.upper()}\n"

        return summary

    def analyze_batch(
        self,
        semantic_features_batch: Optional[torch.Tensor] = None,
        graph_features_batch: Optional[torch.Tensor] = None,
        adversarial_features_batch: Optional[torch.Tensor] = None,
        metadata_features_batch: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """
        Analyze contributions for a batch of samples

        Returns:
            List of analysis results
        """
        batch_size = (
            semantic_features_batch.size(0) if semantic_features_batch is not None
            else graph_features_batch.size(0) if graph_features_batch is not None
            else adversarial_features_batch.size(0) if adversarial_features_batch is not None
            else metadata_features_batch.size(0)
        )

        results = []

        for i in range(batch_size):
            semantic_i = semantic_features_batch[i:i+1] if semantic_features_batch is not None else None
            graph_i = graph_features_batch[i:i+1] if graph_features_batch is not None else None
            adversarial_i = adversarial_features_batch[i:i+1] if adversarial_features_batch is not None else None
            metadata_i = metadata_features_batch[i:i+1] if metadata_features_batch is not None else None

            analysis = self.analyze_contributions(
                semantic_i, graph_i, adversarial_i, metadata_i
            )
            results.append(analysis)

        return results

    def get_aggregate_importance(
        self,
        analyses: List[Dict],
    ) -> Dict[str, float]:
        """
        Aggregate modality importance across multiple samples

        Args:
            analyses: List of analysis results from analyze_batch

        Returns:
            Dictionary with averaged importance scores
        """
        modality_scores = {}

        for analysis in analyses:
            contributions = analysis.get("contributions", {})
            for modality, score in contributions.items():
                if modality not in modality_scores:
                    modality_scores[modality] = []
                modality_scores[modality].append(score)

        # Average scores
        averaged = {
            modality: np.mean(scores)
            for modality, scores in modality_scores.items()
        }

        return averaged


# Example usage
if __name__ == "__main__":
    print("Testing Modality Contribution Analyzer...")

    print("\nModality Analyzer module loaded!")
    print("\nUsage:")
    print("  analyzer = ModalityContributionAnalyzer(fusion_model)")
    print("  analysis = analyzer.analyze_contributions(")
    print("      semantic_features, graph_features, adversarial_features, metadata_features")
    print("  )")
    print("  print(analysis['summary'])")

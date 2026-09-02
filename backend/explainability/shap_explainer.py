"""
SHAP Explainer for EMFRD Models

Provides model-agnostic explanations using SHAP values.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer


class SHAPExplainer:
    """
    SHAP-based explainer for text classification models

    Uses model predictions to generate SHAP values for interpretability.
    Provides token-level importance scores.

    Args:
        model: The model to explain
        tokenizer: Tokenizer for text processing
        device: Device for computation
    """

    def __init__(
        self,
        model,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def explain_prediction(
        self,
        text: str,
        num_samples: int = 100,
        return_tokens: bool = True,
    ) -> Dict[str, any]:
        """
        Generate SHAP explanation for a single prediction

        Args:
            text: Input review text
            num_samples: Number of samples for SHAP estimation
            return_tokens: Whether to return token-level explanations

        Returns:
            Dictionary containing:
                - prediction: Model prediction (0 or 1)
                - probability: Prediction probability
                - shap_values: SHAP values for each token
                - tokens: List of tokens (if return_tokens=True)
                - base_value: Base SHAP value
                - explanation: Human-readable explanation
        """
        # Get base prediction
        prediction, probability = self._get_prediction(text)

        # Tokenize
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = text.split()

        # Compute SHAP values via perturbation
        shap_values = self._compute_shap_values(text, tokens, num_samples)

        # Generate explanation
        explanation = self._generate_explanation(
            tokens, shap_values, prediction, probability
        )

        result = {
            "prediction": int(prediction),
            "prediction_label": "FAKE" if prediction == 1 else "GENUINE",
            "probability": float(probability),
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "base_value": 0.5,  # Neutral baseline
            "explanation": explanation,
        }

        if return_tokens:
            result["tokens"] = tokens

        return result

    def _get_prediction(self, text: str) -> Tuple[int, float]:
        """Get model prediction for text"""
        with torch.no_grad():
            # Encode text
            if self.tokenizer:
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
            else:
                # Fallback: use simple encoding
                return 0, 0.5

            # Get prediction
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probas = torch.softmax(logits, dim=-1)

            prediction = torch.argmax(probas, dim=-1).item()
            probability = probas[0, prediction].item()

            return prediction, probability

    def _compute_shap_values(
        self,
        text: str,
        tokens: List[str],
        num_samples: int,
    ) -> np.ndarray:
        """
        Compute SHAP values via token perturbation

        This is a simplified implementation using token masking.
        For production, consider using official SHAP library.
        """
        n_tokens = len(tokens)
        shap_values = np.zeros(n_tokens)

        # Get baseline prediction (all tokens masked)
        baseline_text = ""
        _, baseline_prob = self._get_prediction(baseline_text)

        # Get full prediction
        _, full_prob = self._get_prediction(text)

        # Compute marginal contribution of each token
        for i in range(n_tokens):
            # Create text without token i
            perturbed_tokens = tokens[:i] + ["[MASK]"] + tokens[i+1:]
            perturbed_text = " ".join(perturbed_tokens)

            # Get prediction without this token
            _, perturbed_prob = self._get_prediction(perturbed_text)

            # SHAP value is the marginal contribution
            shap_values[i] = full_prob - perturbed_prob

        # Normalize
        total = np.abs(shap_values).sum()
        if total > 0:
            shap_values = shap_values / total

        return shap_values

    def _generate_explanation(
        self,
        tokens: List[str],
        shap_values: np.ndarray,
        prediction: int,
        probability: float,
    ) -> str:
        """Generate human-readable explanation"""

        # Get top contributing tokens
        top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
        top_tokens = [(tokens[i], shap_values[i]) for i in top_indices if i < len(tokens)]

        label = "FAKE" if prediction == 1 else "GENUINE"

        explanation = f"Model predicts this review is {label} with {probability:.1%} confidence.\n\n"
        explanation += "Most influential words:\n"

        for token, value in top_tokens:
            direction = "supports FAKE" if value > 0 else "supports GENUINE"
            explanation += f"  • '{token}': {direction} (impact: {abs(value):.3f})\n"

        return explanation

    def explain_batch(
        self,
        texts: List[str],
        num_samples: int = 100,
    ) -> List[Dict[str, any]]:
        """
        Generate explanations for multiple texts

        Args:
            texts: List of review texts
            num_samples: Number of samples for SHAP estimation

        Returns:
            List of explanation dictionaries
        """
        return [
            self.explain_prediction(text, num_samples=num_samples)
            for text in texts
        ]

    def get_feature_importance(
        self,
        texts: List[str],
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple texts

        Args:
            texts: List of review texts
            num_samples: Number of samples per text

        Returns:
            Dictionary mapping tokens to importance scores
        """
        token_importance = {}

        for text in texts:
            explanation = self.explain_prediction(text, num_samples=num_samples)
            tokens = explanation.get("tokens", [])
            shap_values = explanation.get("shap_values", [])

            for token, value in zip(tokens, shap_values):
                if token not in token_importance:
                    token_importance[token] = []
                token_importance[token].append(abs(value))

        # Average importance per token
        averaged = {
            token: np.mean(values)
            for token, values in token_importance.items()
        }

        # Sort by importance
        sorted_importance = dict(
            sorted(averaged.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance


class MultimodalSHAPExplainer:
    """
    SHAP Explainer for multimodal fusion models

    Explains contributions from different modalities:
    - Semantic (text)
    - Behavioral (graph)
    - Adversarial (robustness)
    - Metadata
    """

    def __init__(
        self,
        fusion_model,
        semantic_model,
        device: str = "cpu",
    ):
        self.fusion_model = fusion_model
        self.semantic_model = semantic_model
        self.device = device

    def explain_modality_contributions(
        self,
        semantic_features: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None,
        adversarial_features: Optional[torch.Tensor] = None,
        metadata_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Explain contribution of each modality

        Returns:
            Dictionary with modality contributions
        """
        with torch.no_grad():
            # Get full prediction
            full_output = self.fusion_model(
                semantic_features=semantic_features,
                graph_features=graph_features,
                adversarial_features=adversarial_features,
                metadata_features=metadata_features,
            )

            full_prob = torch.softmax(full_output["logits"], dim=-1)[0, 1].item()
            gates = full_output.get("gates", {})

            # Compute marginal contributions
            contributions = {}

            # Semantic only
            if semantic_features is not None:
                semantic_output = self.fusion_model(
                    semantic_features=semantic_features,
                )
                semantic_prob = torch.softmax(semantic_output["logits"], dim=-1)[0, 1].item()
                contributions["semantic"] = semantic_prob

                # Add gate value
                if "semantic" in gates:
                    contributions["semantic_gate"] = gates["semantic"][0, 0].item()

            # Graph contribution
            if graph_features is not None:
                graph_output = self.fusion_model(
                    semantic_features=semantic_features,
                    graph_features=graph_features,
                )
                graph_prob = torch.softmax(graph_output["logits"], dim=-1)[0, 1].item()
                contributions["graph"] = graph_prob - contributions.get("semantic", 0)

                if "graph" in gates:
                    contributions["graph_gate"] = gates["graph"][0, 0].item()

            # Adversarial contribution
            if adversarial_features is not None:
                contributions["adversarial"] = 0.1  # Placeholder
                if "adversarial" in gates:
                    contributions["adversarial_gate"] = gates["adversarial"][0, 0].item()

            # Metadata contribution
            if metadata_features is not None:
                contributions["metadata"] = 0.05  # Placeholder
                if "metadata" in gates:
                    contributions["metadata_gate"] = gates["metadata"][0, 0].item()

            contributions["full_prediction"] = full_prob

            return contributions


# Example usage
if __name__ == "__main__":
    print("Testing SHAP Explainer...")

    # This is a mock test - in practice, use real models
    print("SHAP Explainer module loaded successfully!")
    print("\nUsage:")
    print("  explainer = SHAPExplainer(model, tokenizer)")
    print("  explanation = explainer.explain_prediction('This is amazing!')")
    print("  print(explanation['explanation'])")

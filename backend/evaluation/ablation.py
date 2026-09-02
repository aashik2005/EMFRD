"""
Ablation Study Framework for EMFRD

Systematically evaluate contribution of each component.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from backend.evaluation.metrics import MetricsCalculator


class AblationStudy:
    """
    Ablation study framework for multimodal models

    Tests performance with different component combinations:
    - All components (full model)
    - Remove one component at a time
    - Use only one component
    - Pairwise combinations

    Args:
        fusion_model: Multimodal fusion model
        component_models: Dictionary of component models
        device: Computation device
    """

    def __init__(
        self,
        fusion_model,
        component_models: Dict[str, any],
        device: str = "cpu",
    ):
        self.fusion_model = fusion_model
        self.component_models = component_models
        self.device = device
        self.metrics_calc = MetricsCalculator()

        # Define ablation configurations
        self.configurations = self._define_configurations()

    def _define_configurations(self) -> List[Dict]:
        """
        Define all ablation configurations to test

        Returns:
            List of configuration dictionaries
        """
        configs = []

        # Full model (all modalities)
        configs.append({
            "name": "Full Model",
            "description": "All modalities enabled",
            "modalities": {
                "semantic": True,
                "graph": True,
                "adversarial": True,
                "metadata": True,
            },
        })

        # Ablations: Remove one modality
        ablations = [
            ("No Semantic", "semantic"),
            ("No Graph", "graph"),
            ("No Adversarial", "adversarial"),
            ("No Metadata", "metadata"),
        ]

        for name, removed_modality in ablations:
            config = {
                "name": name,
                "description": f"All modalities except {removed_modality}",
                "modalities": {
                    "semantic": removed_modality != "semantic",
                    "graph": removed_modality != "graph",
                    "adversarial": removed_modality != "adversarial",
                    "metadata": removed_modality != "metadata",
                },
            }
            configs.append(config)

        # Single modality tests
        single_modalities = [
            ("Semantic Only", "semantic"),
            ("Graph Only", "graph"),
            ("Adversarial Only", "adversarial"),
            ("Metadata Only", "metadata"),
        ]

        for name, modality in single_modalities:
            config = {
                "name": name,
                "description": f"Only {modality} modality",
                "modalities": {
                    "semantic": modality == "semantic",
                    "graph": modality == "graph",
                    "adversarial": modality == "adversarial",
                    "metadata": modality == "metadata",
                },
            }
            configs.append(config)

        # Pairwise combinations
        pairs = [
            ("Semantic + Graph", ["semantic", "graph"]),
            ("Semantic + Adversarial", ["semantic", "adversarial"]),
            ("Semantic + Metadata", ["semantic", "metadata"]),
            ("Graph + Adversarial", ["graph", "adversarial"]),
        ]

        for name, active_modalities in pairs:
            config = {
                "name": name,
                "description": f"Only {' and '.join(active_modalities)}",
                "modalities": {
                    "semantic": "semantic" in active_modalities,
                    "graph": "graph" in active_modalities,
                    "adversarial": "adversarial" in active_modalities,
                    "metadata": "metadata" in active_modalities,
                },
            }
            configs.append(config)

        return configs

    def run_ablation(
        self,
        test_loader,
        extract_features_fn,
        save_results: bool = True,
        results_dir: Optional[Path] = None,
    ) -> Dict[str, any]:
        """
        Run complete ablation study

        Args:
            test_loader: DataLoader for test set
            extract_features_fn: Function to extract features from batch
            save_results: Whether to save results to file
            results_dir: Directory to save results

        Returns:
            Dictionary with ablation results
        """
        print("="*70)
        print("ABLATION STUDY: EMFRD Components")
        print("="*70)

        results = {
            "configurations": [],
            "summary": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Test each configuration
        for i, config in enumerate(self.configurations, 1):
            print(f"\n[{i}/{len(self.configurations)}] Testing: {config['name']}")
            print(f"    {config['description']}")

            metrics = self._evaluate_configuration(
                config,
                test_loader,
                extract_features_fn,
            )

            config_result = {
                "name": config["name"],
                "description": config["description"],
                "modalities": config["modalities"],
                "metrics": metrics,
            }

            results["configurations"].append(config_result)

            # Print results
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")

        # Generate summary
        results["summary"] = self._generate_summary(results["configurations"])

        # Print summary table
        self._print_summary_table(results["configurations"])

        # Save results
        if save_results:
            if results_dir is None:
                results_dir = Path("experiments/ablation")
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = results_dir / f"ablation_study_{timestamp}.json"

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n✓ Results saved to: {results_path}")

        return results

    def _evaluate_configuration(
        self,
        config: Dict,
        test_loader,
        extract_features_fn,
    ) -> Dict[str, float]:
        """Evaluate a single ablation configuration"""
        self.fusion_model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                labels = batch["labels"].to(self.device)

                # Extract all features
                features = extract_features_fn(batch)

                # Apply ablation: Set disabled modalities to None
                ablated_features = {
                    "semantic": features["semantic"] if config["modalities"]["semantic"] else None,
                    "graph": features["graph"] if config["modalities"]["graph"] else None,
                    "adversarial": features["adversarial"] if config["modalities"]["adversarial"] else None,
                    "metadata": features["metadata"] if config["modalities"]["metadata"] else None,
                }

                # Check if at least one modality is enabled
                if not any(config["modalities"].values()):
                    # No modalities enabled - random predictions
                    batch_size = labels.size(0)
                    predictions = torch.randint(0, 2, (batch_size,), device=self.device)
                    probabilities = torch.ones(batch_size, device=self.device) * 0.5
                else:
                    try:
                        # Forward pass
                        outputs = self.fusion_model(
                            semantic_features=ablated_features["semantic"],
                            graph_features=ablated_features["graph"],
                            adversarial_features=ablated_features["adversarial"],
                            metadata_features=ablated_features["metadata"],
                        )

                        logits = outputs["logits"]
                        probas = torch.softmax(logits, dim=-1)
                        predictions = torch.argmax(logits, dim=-1)
                        probabilities = probas[:, 1]  # Probability of fake class

                    except Exception as e:
                        # Fallback if configuration not supported
                        print(f"      Warning: {e}")
                        batch_size = labels.size(0)
                        predictions = torch.randint(0, 2, (batch_size,), device=self.device)
                        probabilities = torch.ones(batch_size, device=self.device) * 0.5

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all(
            all_labels,
            all_predictions,
            all_probabilities,
        )

        return metrics

    def _generate_summary(self, configurations: List[Dict]) -> Dict:
        """Generate summary statistics"""
        summary = {
            "best_configuration": None,
            "worst_configuration": None,
            "full_model_rank": None,
            "modality_importance": {},
        }

        # Find best and worst
        sorted_configs = sorted(
            configurations,
            key=lambda x: x["metrics"]["f1"],
            reverse=True,
        )

        summary["best_configuration"] = sorted_configs[0]["name"]
        summary["worst_configuration"] = sorted_configs[-1]["name"]

        # Find full model rank
        for i, config in enumerate(sorted_configs, 1):
            if config["name"] == "Full Model":
                summary["full_model_rank"] = i
                break

        # Compute modality importance
        full_model_f1 = next(
            c["metrics"]["f1"] for c in configurations if c["name"] == "Full Model"
        )

        importance = {}
        for modality in ["semantic", "graph", "adversarial", "metadata"]:
            no_modality_name = f"No {modality.capitalize()}"
            no_modality_config = next(
                (c for c in configurations if c["name"] == no_modality_name),
                None
            )

            if no_modality_config:
                no_modality_f1 = no_modality_config["metrics"]["f1"]
                importance[modality] = full_model_f1 - no_modality_f1

        summary["modality_importance"] = importance

        return summary

    def _print_summary_table(self, configurations: List[Dict]):
        """Print formatted summary table"""
        print("\n" + "="*70)
        print("ABLATION RESULTS SUMMARY")
        print("="*70)

        # Sort by F1 score
        sorted_configs = sorted(
            configurations,
            key=lambda x: x["metrics"]["f1"],
            reverse=True,
        )

        # Print header
        print(f"\n{'Rank':<6} {'Configuration':<30} {'Acc':<10} {'P':<10} {'R':<10} {'F1':<10}")
        print("-" * 70)

        # Print rows
        for rank, config in enumerate(sorted_configs, 1):
            metrics = config["metrics"]
            print(
                f"{rank:<6} "
                f"{config['name']:<30} "
                f"{metrics['accuracy']:<10.4f} "
                f"{metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} "
                f"{metrics['f1']:<10.4f}"
            )

        print("=" * 70)


class RobustnessEvaluator:
    """
    Evaluate model robustness to perturbations

    Tests:
    - Adversarial word substitutions
    - Character-level noise
    - Sentence reordering
    - Length variations
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_calc = MetricsCalculator()

    def evaluate_robustness(
        self,
        test_texts: List[str],
        test_labels: List[int],
        perturbations: List[str] = None,
    ) -> Dict[str, any]:
        """
        Evaluate robustness across multiple perturbation types

        Args:
            test_texts: Original test texts
            test_labels: True labels
            perturbations: List of perturbation types to test

        Returns:
            Dictionary with robustness results
        """
        if perturbations is None:
            perturbations = [
                "original",
                "word_swap",
                "char_noise",
                "sentence_reorder",
            ]

        results = {}

        for perturb_type in perturbations:
            print(f"\nTesting robustness: {perturb_type}")

            if perturb_type == "original":
                perturbed_texts = test_texts
            elif perturb_type == "word_swap":
                perturbed_texts = [self._word_swap(text) for text in test_texts]
            elif perturb_type == "char_noise":
                perturbed_texts = [self._add_char_noise(text) for text in test_texts]
            elif perturb_type == "sentence_reorder":
                perturbed_texts = [self._reorder_sentences(text) for text in test_texts]
            else:
                continue

            # Evaluate on perturbed data
            predictions = self._predict_batch(perturbed_texts)
            metrics = self.metrics_calc.calculate_all(test_labels, predictions)

            results[perturb_type] = metrics

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")

        return results

    def _predict_batch(self, texts: List[str]) -> List[int]:
        """Make predictions for a batch of texts"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                prediction = torch.argmax(logits, dim=-1).item()
                predictions.append(prediction)

        return predictions

    def _word_swap(self, text: str, swap_prob: float = 0.1) -> str:
        """Swap random words"""
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(max(1, int(len(words) * swap_prob))):
            if len(words) >= 2:
                i, j = np.random.choice(len(words), size=2, replace=False)
                words[i], words[j] = words[j], words[i]

        return " ".join(words)

    def _add_char_noise(self, text: str, noise_prob: float = 0.05) -> str:
        """Add character-level noise"""
        chars = list(text)
        for i in range(len(chars)):
            if np.random.random() < noise_prob and chars[i].isalnum():
                # Random character substitution
                chars[i] = chr(ord(chars[i]) + np.random.randint(-2, 3))

        return "".join(chars)

    def _reorder_sentences(self, text: str) -> str:
        """Randomly reorder sentences"""
        sentences = text.split(". ")
        if len(sentences) > 1:
            np.random.shuffle(sentences)
        return ". ".join(sentences)


# Example usage
if __name__ == "__main__":
    print("Ablation Study Framework loaded!")
    print("\nUsage:")
    print("  study = AblationStudy(fusion_model, component_models, device)")
    print("  results = study.run_ablation(test_loader, extract_features_fn)")

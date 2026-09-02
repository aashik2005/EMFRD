"""
Counterfactual Generation for EMFRD

Generates counterfactual examples: minimal changes to flip prediction.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import re


class CounterfactualGenerator:
    """
    Generate counterfactual explanations for review classification

    Finds minimal perturbations that would flip the prediction.

    Args:
        model: Classification model
        tokenizer: Text tokenizer
        device: Computation device
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Common substitutions for counterfactuals
        self.fake_indicators = [
            "amazing", "incredible", "best ever", "perfect", "awesome",
            "fantastic", "outstanding", "excellent", "superb", "wonderful",
            "life-changing", "must buy", "highly recommend", "love it",
        ]

        self.genuine_indicators = [
            "good", "decent", "okay", "fine", "adequate", "satisfactory",
            "works", "useful", "practical", "reasonable", "average",
            "as expected", "fair", "acceptable",
        ]

        self.suspicious_patterns = [
            r"!!!+",  # Multiple exclamations
            r"\b([A-Z]+)\b",  # All caps words
            r"(.)\1{2,}",  # Repeated characters
            r"best (ever|product|item)",
            r"must buy",
            r"highly recommend",
            r"(amazing|incredible|perfect)",
        ]

    def generate_counterfactual(
        self,
        text: str,
        target_class: Optional[int] = None,
        max_changes: int = 5,
        beam_size: int = 5,
    ) -> Dict[str, any]:
        """
        Generate counterfactual explanation

        Args:
            text: Original review text
            target_class: Target class to flip to (None = opposite of current)
            max_changes: Maximum number of token changes
            beam_size: Number of candidates to explore

        Returns:
            Dictionary containing:
                - original_text: Original review
                - original_prediction: Original prediction
                - counterfactual_text: Modified review
                - counterfactual_prediction: New prediction
                - changes: List of changes made
                - num_changes: Number of changes
                - success: Whether flip was successful
        """
        # Get original prediction
        orig_pred, orig_prob = self._get_prediction(text)

        if target_class is None:
            target_class = 1 - orig_pred

        # Try multiple strategies
        strategies = [
            self._strategy_word_substitution,
            self._strategy_remove_suspicious_patterns,
            self._strategy_modify_intensity,
            self._strategy_add_qualifiers,
        ]

        best_counterfactual = None
        min_changes = max_changes + 1

        for strategy in strategies:
            cf_text, changes = strategy(text, target_class, max_changes)

            # Check if successful
            cf_pred, cf_prob = self._get_prediction(cf_text)

            if cf_pred == target_class and len(changes) < min_changes:
                best_counterfactual = {
                    "original_text": text,
                    "original_prediction": int(orig_pred),
                    "original_probability": float(orig_prob),
                    "counterfactual_text": cf_text,
                    "counterfactual_prediction": int(cf_pred),
                    "counterfactual_probability": float(cf_prob),
                    "changes": changes,
                    "num_changes": len(changes),
                    "success": True,
                    "strategy": strategy.__name__,
                }
                min_changes = len(changes)

        if best_counterfactual is None:
            # Return best effort
            cf_text = text
            changes = []
            best_counterfactual = {
                "original_text": text,
                "original_prediction": int(orig_pred),
                "original_probability": float(orig_prob),
                "counterfactual_text": cf_text,
                "counterfactual_prediction": int(orig_pred),
                "counterfactual_probability": float(orig_prob),
                "changes": changes,
                "num_changes": 0,
                "success": False,
                "strategy": "none",
            }

        return best_counterfactual

    def _get_prediction(self, text: str) -> Tuple[int, float]:
        """Get model prediction"""
        with torch.no_grad():
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
            probas = torch.softmax(logits, dim=-1)

            prediction = torch.argmax(probas, dim=-1).item()
            probability = probas[0, prediction].item()

            return prediction, probability

    def _strategy_word_substitution(
        self,
        text: str,
        target_class: int,
        max_changes: int,
    ) -> Tuple[str, List[Dict]]:
        """
        Strategy 1: Substitute extreme words with moderate ones

        If flipping to GENUINE (0): Replace fake indicators with genuine ones
        If flipping to FAKE (1): Replace genuine indicators with fake ones
        """
        changes = []
        modified_text = text

        if target_class == 0:  # Flip to GENUINE
            # Replace fake indicators
            for fake_word in self.fake_indicators:
                if fake_word.lower() in modified_text.lower() and len(changes) < max_changes:
                    # Find genuine replacement
                    genuine_word = np.random.choice(self.genuine_indicators)

                    # Case-preserving replacement
                    pattern = re.compile(re.escape(fake_word), re.IGNORECASE)
                    new_text = pattern.sub(genuine_word, modified_text, count=1)

                    if new_text != modified_text:
                        changes.append({
                            "type": "substitution",
                            "original": fake_word,
                            "replacement": genuine_word,
                            "position": modified_text.lower().find(fake_word.lower()),
                        })
                        modified_text = new_text

        else:  # Flip to FAKE
            # Replace genuine indicators with fake ones
            for genuine_word in self.genuine_indicators:
                if genuine_word.lower() in modified_text.lower() and len(changes) < max_changes:
                    fake_word = np.random.choice(self.fake_indicators)

                    pattern = re.compile(re.escape(genuine_word), re.IGNORECASE)
                    new_text = pattern.sub(fake_word, modified_text, count=1)

                    if new_text != modified_text:
                        changes.append({
                            "type": "substitution",
                            "original": genuine_word,
                            "replacement": fake_word,
                            "position": modified_text.lower().find(genuine_word.lower()),
                        })
                        modified_text = new_text

        return modified_text, changes

    def _strategy_remove_suspicious_patterns(
        self,
        text: str,
        target_class: int,
        max_changes: int,
    ) -> Tuple[str, List[Dict]]:
        """
        Strategy 2: Remove suspicious patterns (to flip to GENUINE)
        """
        if target_class != 0:  # Only works for flipping to GENUINE
            return text, []

        changes = []
        modified_text = text

        for pattern in self.suspicious_patterns:
            if len(changes) >= max_changes:
                break

            matches = list(re.finditer(pattern, modified_text, re.IGNORECASE))
            for match in matches:
                if len(changes) >= max_changes:
                    break

                original = match.group(0)

                # Remove or tone down
                if pattern == r"!!!+":
                    replacement = "."
                elif pattern == r"\b([A-Z]+)\b":
                    replacement = original.lower()
                elif pattern == r"(.)\1{2,}":
                    replacement = match.group(1)
                else:
                    replacement = ""

                modified_text = modified_text[:match.start()] + replacement + modified_text[match.end():]

                changes.append({
                    "type": "pattern_removal",
                    "original": original,
                    "replacement": replacement,
                    "position": match.start(),
                })

        return modified_text, changes

    def _strategy_modify_intensity(
        self,
        text: str,
        target_class: int,
        max_changes: int,
    ) -> Tuple[str, List[Dict]]:
        """
        Strategy 3: Modify intensity of expressions
        """
        changes = []
        modified_text = text

        # Intensity modifiers
        high_intensity = ["very", "extremely", "absolutely", "totally", "completely"]
        low_intensity = ["somewhat", "fairly", "relatively", "quite", "pretty"]

        if target_class == 0:  # Reduce intensity for GENUINE
            for word in high_intensity:
                if word in modified_text.lower() and len(changes) < max_changes:
                    # Remove intensifier
                    pattern = re.compile(r'\b' + re.escape(word) + r'\s+', re.IGNORECASE)
                    new_text = pattern.sub('', modified_text, count=1)

                    if new_text != modified_text:
                        changes.append({
                            "type": "intensity_reduction",
                            "original": word,
                            "replacement": "",
                        })
                        modified_text = new_text

        else:  # Increase intensity for FAKE
            words = modified_text.split()
            for i, word in enumerate(words):
                if len(changes) >= max_changes:
                    break

                # Add intensifier before adjectives
                if word.lower() in ["good", "nice", "great", "cool"]:
                    intensifier = np.random.choice(high_intensity)
                    words[i] = f"{intensifier} {word}"
                    changes.append({
                        "type": "intensity_increase",
                        "original": word,
                        "replacement": f"{intensifier} {word}",
                    })

            modified_text = " ".join(words)

        return modified_text, changes

    def _strategy_add_qualifiers(
        self,
        text: str,
        target_class: int,
        max_changes: int,
    ) -> Tuple[str, List[Dict]]:
        """
        Strategy 4: Add qualifying phrases to make more genuine
        """
        if target_class != 0:  # Only for flipping to GENUINE
            return text, []

        changes = []

        qualifiers = [
            "In my experience, ",
            "I found that ",
            "Based on my usage, ",
            "After testing it, ",
        ]

        # Add qualifier to beginning if not already present
        if not any(q.lower() in text.lower() for q in qualifiers):
            qualifier = np.random.choice(qualifiers)
            modified_text = qualifier + text
            changes.append({
                "type": "qualifier_added",
                "original": "",
                "replacement": qualifier,
                "position": 0,
            })
        else:
            modified_text = text

        return modified_text, changes

    def generate_multiple_counterfactuals(
        self,
        text: str,
        num_counterfactuals: int = 3,
        max_changes: int = 5,
    ) -> List[Dict]:
        """
        Generate multiple diverse counterfactuals

        Args:
            text: Original review text
            num_counterfactuals: Number of counterfactuals to generate
            max_changes: Maximum changes per counterfactual

        Returns:
            List of counterfactual dictionaries
        """
        counterfactuals = []

        # Get original prediction
        orig_pred, _ = self._get_prediction(text)
        target_class = 1 - orig_pred

        # Try with different strategies and max_changes
        for i in range(num_counterfactuals):
            max_ch = min(max_changes, i + 1)  # Vary number of changes
            cf = self.generate_counterfactual(text, target_class, max_ch)

            # Only add if successful and unique
            if cf["success"]:
                is_unique = all(
                    cf["counterfactual_text"] != existing["counterfactual_text"]
                    for existing in counterfactuals
                )
                if is_unique:
                    counterfactuals.append(cf)

        return counterfactuals


# Example usage
if __name__ == "__main__":
    print("Testing Counterfactual Generator...")

    # Mock example
    print("\nCounterfactual Generator module loaded!")
    print("\nUsage:")
    print("  generator = CounterfactualGenerator(model, tokenizer)")
    print("  cf = generator.generate_counterfactual('This is AMAZING!!!')")
    print("  print(f'Original: {cf[\"original_text\"]}')")
    print("  print(f'Counterfactual: {cf[\"counterfactual_text\"]}')")

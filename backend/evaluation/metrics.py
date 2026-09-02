"""
Metrics calculation for EMFRD

IMPORTANT: All metrics are calculated from ACTUAL predictions,
never hard-coded or fabricated.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import json


@dataclass
class EvaluationResults:
    """
    Container for evaluation metrics

    All values are calculated from actual predictions
    """

    # Core metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None

    # Detailed metrics
    specificity: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None

    # Confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Additional info
    num_samples: int = 0
    num_fake: int = 0
    num_genuine: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def print_summary(self, model_name: str = "Model"):
        """Print formatted summary"""
        print(f"\n{'='*60}")
        print(f"{model_name} Evaluation Results")
        print(f"{'='*60}")
        print(f"Dataset: {self.num_samples} samples ({self.num_fake} fake, {self.num_genuine} genuine)")
        print(f"\nCore Metrics:")
        print(f"  Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"  Precision: {self.precision:.4f} ({self.precision*100:.2f}%)")
        print(f"  Recall:    {self.recall:.4f} ({self.recall*100:.2f}%)")
        print(f"  F1 Score:  {self.f1:.4f} ({self.f1*100:.2f}%)")

        if self.roc_auc is not None:
            print(f"  ROC-AUC:   {self.roc_auc:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {self.true_positives}")
        print(f"  True Negatives:  {self.true_negatives}")
        print(f"  False Positives: {self.false_positives}")
        print(f"  False Negatives: {self.false_negatives}")

        if self.specificity is not None:
            print(f"\nAdditional Metrics:")
            print(f"  Specificity: {self.specificity:.4f}")
            print(f"  FPR: {self.false_positive_rate:.4f}")
            print(f"  FNR: {self.false_negative_rate:.4f}")

        print(f"{'='*60}\n")


class MetricsCalculator:
    """
    Calculate evaluation metrics from predictions

    IMPORTANT: This class ONLY calculates metrics from actual predictions.
    It does NOT hard-code or fabricate results.
    """

    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "binary",
    ) -> EvaluationResults:
        """
        Calculate comprehensive metrics

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_proba: Predicted probabilities (optional, for ROC-AUC)
            average: Averaging method for metrics

        Returns:
            EvaluationResults object
        """
        # Basic validation
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        # Core metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        # ROC-AUC (requires probabilities)
        roc_auc = None
        if y_proba is not None:
            try:
                # For binary classification, use probability of positive class
                if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Dataset statistics
        num_samples = len(y_true)
        num_fake = int(np.sum(y_true == 1))
        num_genuine = int(np.sum(y_true == 0))

        return EvaluationResults(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc) if roc_auc is not None else None,
            specificity=float(specificity),
            false_positive_rate=float(fpr),
            false_negative_rate=float(fnr),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            num_samples=num_samples,
            num_fake=num_fake,
            num_genuine=num_genuine,
        )

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None,
    ) -> str:
        """
        Get detailed classification report

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            target_names: Class names (default: ["Genuine", "Fake"])

        Returns:
            Classification report string
        """
        if target_names is None:
            target_names = ["Genuine", "Fake"]

        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
        )

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Get confusion matrix

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix (2x2 for binary classification)
        """
        return confusion_matrix(y_true, y_pred, labels=[0, 1])

    @staticmethod
    def compare_with_paper(
        our_results: EvaluationResults,
        paper_results: Dict[str, float],
        model_name: str = "Model",
    ):
        """
        Compare our results with paper reference results

        Args:
            our_results: Our evaluation results
            paper_results: Paper reference results dict
            model_name: Model name for display
        """
        print(f"\n{'='*70}")
        print(f"{model_name}: Our Results vs Paper Reference")
        print(f"{'='*70}")
        print(f"{'Metric':<15} {'Our Result':>15} {'Paper Ref':>15} {'Difference':>15}")
        print(f"{'-'*70}")

        metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in metrics:
            our_val = getattr(our_results, metric, None)
            paper_val = paper_results.get(metric)

            if our_val is not None and paper_val is not None:
                diff = (our_val - paper_val) * 100  # Percentage point difference
                sign = "+" if diff > 0 else ""
                print(f"{metric.capitalize():<15} {our_val*100:>14.2f}% {paper_val*100:>14.2f}% {sign}{diff:>13.2f}pp")

        print(f"{'='*70}\n")

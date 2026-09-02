"""
Text preprocessing for EMFRD

IMPORTANT: Minimal preprocessing for RoBERTa
RoBERTa is pretrained on raw text, so we preserve most semantic information
"""
import re
from typing import List


class TextPreprocessor:
    """
    Text preprocessor for review data

    Philosophy:
    - Minimal preprocessing to preserve semantic information
    - RoBERTa handles tokenization and subword segmentation
    - Focus on cleaning artifacts, not linguistic normalization
    """

    def __init__(
        self,
        lowercase: bool = False,  # RoBERTa is case-sensitive
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize preprocessor

        Args:
            lowercase: Convert to lowercase (NOT recommended for RoBERTa)
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            normalize_whitespace: Normalize excessive whitespace
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace

    def preprocess(self, text: str) -> str:
        """
        Preprocess a single review text

        Args:
            text: Raw review text

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'&[a-z]+;', ' ', text)  # HTML entities

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', ' ', text)

        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', ' ', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        # Lowercase (optional, NOT recommended for RoBERTa)
        if self.lowercase:
            text = text.lower()

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess batch of texts

        Args:
            texts: List of raw texts

        Returns:
            List of cleaned texts
        """
        return [self.preprocess(text) for text in texts]

    def is_valid_review(self, text: str, min_length: int = 10, max_length: int = 5000) -> bool:
        """
        Check if review is valid

        Args:
            text: Review text
            min_length: Minimum character length
            max_length: Maximum character length

        Returns:
            True if valid
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()

        if len(text) < min_length or len(text) > max_length:
            return False

        # Check if text is mostly non-alphanumeric
        alphanum_count = sum(c.isalnum() for c in text)
        if alphanum_count / len(text) < 0.5:
            return False

        return True

    def filter_valid_reviews(
        self,
        texts: List[str],
        labels: List[int],
        min_length: int = 10,
        max_length: int = 5000,
    ) -> tuple:
        """
        Filter out invalid reviews

        Args:
            texts: List of review texts
            labels: List of labels
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            Tuple of (filtered_texts, filtered_labels)
        """
        valid_indices = [
            i for i, text in enumerate(texts)
            if self.is_valid_review(text, min_length, max_length)
        ]

        filtered_texts = [texts[i] for i in valid_indices]
        filtered_labels = [labels[i] for i in valid_indices]

        print(f"Filtered reviews: {len(texts)} -> {len(filtered_texts)} "
              f"(removed {len(texts) - len(filtered_texts)})")

        return filtered_texts, filtered_labels

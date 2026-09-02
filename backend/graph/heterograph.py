"""
Heterogeneous Graph Construction for EMFRD

Constructs User-Review-Product graphs for behavioral analysis.

Graph Schema:
    USER --writes--> REVIEW
    REVIEW --about--> PRODUCT
    USER --reviews--> PRODUCT (derived from above)

Node Types:
    - USER: Reviewers
    - REVIEW: Individual reviews
    - PRODUCT: Products being reviewed

Edge Types:
    - (user, writes, review)
    - (review, about, product)
    - (user, reviews, product)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings

try:
    import dgl
    import torch
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not installed. Graph features will be unavailable. Install with: pip install dgl")


class GraphFeatures:
    """
    Container for graph-based behavioral features
    """

    def __init__(
        self,
        review_id: str,
        user_features: Optional[Dict] = None,
        product_features: Optional[Dict] = None,
        graph_embedding: Optional[np.ndarray] = None,
    ):
        self.review_id = review_id
        self.user_features = user_features or {}
        self.product_features = product_features or {}
        self.graph_embedding = graph_embedding

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "review_id": self.review_id,
            "user_features": self.user_features,
            "product_features": self.product_features,
            "has_graph_embedding": self.graph_embedding is not None,
        }


class HeterographBuilder:
    """
    Builds heterogeneous graphs from review data

    Args:
        reviews: List of ReviewRecord objects
        min_reviews_per_user: Minimum reviews per user to include (filter spam)
        min_reviews_per_product: Minimum reviews per product to include
    """

    def __init__(
        self,
        reviews: List,
        min_reviews_per_user: int = 1,
        min_reviews_per_product: int = 1,
    ):
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for graph construction. Install with: pip install dgl")

        self.reviews = reviews
        self.min_reviews_per_user = min_reviews_per_user
        self.min_reviews_per_product = min_reviews_per_product

        # Check if data has required fields
        self.has_user_id = any(r.user_id is not None for r in reviews)
        self.has_product_id = any(r.product_id is not None for r in reviews)

        if not self.has_user_id or not self.has_product_id:
            raise ValueError(
                "Dataset does not contain user_id and product_id required for graph construction. "
                "Consider using FraudAmazon dataset for graph experiments."
            )

        # Build mappings
        self.user_to_idx = {}
        self.product_to_idx = {}
        self.review_to_idx = {}

        self.graph = None
        self.node_features = {}

    def build(self) -> dgl.DGLGraph:
        """
        Build heterogeneous graph

        Returns:
            DGL heterograph
        """
        print("Building heterogeneous graph...")

        # Filter reviews
        valid_reviews = [
            r for r in self.reviews
            if r.user_id is not None and r.product_id is not None
        ]

        print(f"Valid reviews for graph: {len(valid_reviews)} / {len(self.reviews)}")

        # Count reviews per user/product
        user_review_counts = defaultdict(int)
        product_review_counts = defaultdict(int)

        for r in valid_reviews:
            user_review_counts[r.user_id] += 1
            product_review_counts[r.product_id] += 1

        # Filter by minimum review counts
        valid_users = {
            u for u, count in user_review_counts.items()
            if count >= self.min_reviews_per_user
        }
        valid_products = {
            p for p, count in product_review_counts.items()
            if count >= self.min_reviews_per_product
        }

        print(f"Valid users: {len(valid_users)}")
        print(f"Valid products: {len(valid_products)}")

        # Filter reviews to only valid users/products
        filtered_reviews = [
            r for r in valid_reviews
            if r.user_id in valid_users and r.product_id in valid_products
        ]

        print(f"Filtered reviews: {len(filtered_reviews)}")

        if len(filtered_reviews) == 0:
            raise ValueError("No valid reviews after filtering. Try lowering min_reviews thresholds.")

        # Create node mappings
        users = sorted(list(valid_users))
        products = sorted(list(valid_products))

        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        self.product_to_idx = {p: idx for idx, p in enumerate(products)}
        self.review_to_idx = {r.review_id: idx for idx, r in enumerate(filtered_reviews)}

        # Build edge lists
        user_writes_review_edges = ([], [])  # (user_ids, review_ids)
        review_about_product_edges = ([], [])  # (review_ids, product_ids)
        user_reviews_product_edges = ([], [])  # (user_ids, product_ids)

        for r in filtered_reviews:
            user_idx = self.user_to_idx[r.user_id]
            product_idx = self.product_to_idx[r.product_id]
            review_idx = self.review_to_idx[r.review_id]

            # User writes review
            user_writes_review_edges[0].append(user_idx)
            user_writes_review_edges[1].append(review_idx)

            # Review about product
            review_about_product_edges[0].append(review_idx)
            review_about_product_edges[1].append(product_idx)

            # User reviews product (derived)
            user_reviews_product_edges[0].append(user_idx)
            user_reviews_product_edges[1].append(product_idx)

        # Create heterograph
        graph_data = {
            ('user', 'writes', 'review'): (
                torch.tensor(user_writes_review_edges[0]),
                torch.tensor(user_writes_review_edges[1])
            ),
            ('review', 'about', 'product'): (
                torch.tensor(review_about_product_edges[0]),
                torch.tensor(review_about_product_edges[1])
            ),
            ('user', 'reviews', 'product'): (
                torch.tensor(user_reviews_product_edges[0]),
                torch.tensor(user_reviews_product_edges[1])
            ),
        }

        self.graph = dgl.heterograph(graph_data)

        print(f"\nGraph statistics:")
        print(f"  Nodes: {self.graph.num_nodes()} total")
        print(f"    Users: {self.graph.num_nodes('user')}")
        print(f"    Reviews: {self.graph.num_nodes('review')}")
        print(f"    Products: {self.graph.num_nodes('product')}")
        print(f"  Edges: {self.graph.num_edges()} total")
        print(f"    User writes review: {self.graph.num_edges('writes')}")
        print(f"    Review about product: {self.graph.num_edges('about')}")
        print(f"    User reviews product: {self.graph.num_edges('reviews')}")

        # Add node features
        self._add_node_features(filtered_reviews)

        return self.graph

    def _add_node_features(self, reviews: List):
        """Add features to graph nodes"""
        print("\nAdding node features...")

        # Review node features
        review_labels = torch.zeros(len(reviews), dtype=torch.long)
        review_ratings = torch.zeros(len(reviews), dtype=torch.float)

        for idx, r in enumerate(reviews):
            review_labels[idx] = r.label
            review_ratings[idx] = r.rating if r.rating is not None else 0.0

        self.graph.nodes['review'].data['label'] = review_labels
        self.graph.nodes['review'].data['rating'] = review_ratings

        # User node features (behavioral)
        num_users = self.graph.num_nodes('user')
        user_review_counts = torch.zeros(num_users, dtype=torch.float)
        user_fake_ratios = torch.zeros(num_users, dtype=torch.float)
        user_avg_ratings = torch.zeros(num_users, dtype=torch.float)

        user_reviews = defaultdict(list)
        for r in reviews:
            user_idx = self.user_to_idx[r.user_id]
            user_reviews[user_idx].append(r)

        for user_idx, user_review_list in user_reviews.items():
            user_review_counts[user_idx] = len(user_review_list)
            fake_count = sum(1 for r in user_review_list if r.label == 1)
            user_fake_ratios[user_idx] = fake_count / len(user_review_list) if user_review_list else 0.0
            ratings = [r.rating for r in user_review_list if r.rating is not None]
            user_avg_ratings[user_idx] = np.mean(ratings) if ratings else 0.0

        self.graph.nodes['user'].data['review_count'] = user_review_counts
        self.graph.nodes['user'].data['fake_ratio'] = user_fake_ratios
        self.graph.nodes['user'].data['avg_rating'] = user_avg_ratings

        # Product node features
        num_products = self.graph.num_nodes('product')
        product_review_counts = torch.zeros(num_products, dtype=torch.float)
        product_fake_ratios = torch.zeros(num_products, dtype=torch.float)
        product_avg_ratings = torch.zeros(num_products, dtype=torch.float)

        product_reviews = defaultdict(list)
        for r in reviews:
            product_idx = self.product_to_idx[r.product_id]
            product_reviews[product_idx].append(r)

        for product_idx, product_review_list in product_reviews.items():
            product_review_counts[product_idx] = len(product_review_list)
            fake_count = sum(1 for r in product_review_list if r.label == 1)
            product_fake_ratios[product_idx] = fake_count / len(product_review_list) if product_review_list else 0.0
            ratings = [r.rating for r in product_review_list if r.rating is not None]
            product_avg_ratings[product_idx] = np.mean(ratings) if ratings else 0.0

        self.graph.nodes['product'].data['review_count'] = product_review_counts
        self.graph.nodes['product'].data['fake_ratio'] = product_fake_ratios
        self.graph.nodes['product'].data['avg_rating'] = product_avg_ratings

        print("Node features added successfully")

    def extract_review_features(self, review_id: str) -> Optional[GraphFeatures]:
        """
        Extract graph-based features for a specific review

        Args:
            review_id: Review ID

        Returns:
            GraphFeatures object or None if review not in graph
        """
        if review_id not in self.review_to_idx:
            return None

        review_idx = self.review_to_idx[review_id]

        # Find user and product for this review
        review_node = torch.tensor([review_idx])

        # Get user (incoming 'writes' edges to this review)
        user_ids, _ = self.graph.in_edges(review_node, etype='writes')
        user_idx = user_ids[0].item() if len(user_ids) > 0 else None

        # Get product (outgoing 'about' edges from this review)
        _, product_ids = self.graph.out_edges(review_node, etype='about')
        product_idx = product_ids[0].item() if len(product_ids) > 0 else None

        # Extract features
        user_features = {}
        if user_idx is not None:
            user_features = {
                'review_count': float(self.graph.nodes['user'].data['review_count'][user_idx]),
                'fake_ratio': float(self.graph.nodes['user'].data['fake_ratio'][user_idx]),
                'avg_rating': float(self.graph.nodes['user'].data['avg_rating'][user_idx]),
            }

        product_features = {}
        if product_idx is not None:
            product_features = {
                'review_count': float(self.graph.nodes['product'].data['review_count'][product_idx]),
                'fake_ratio': float(self.graph.nodes['product'].data['fake_ratio'][product_idx]),
                'avg_rating': float(self.graph.nodes['product'].data['avg_rating'][product_idx]),
            }

        return GraphFeatures(
            review_id=review_id,
            user_features=user_features,
            product_features=product_features,
        )

    def get_suspicious_users(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get users with high fake review ratios

        Args:
            threshold: Minimum fake ratio to consider suspicious

        Returns:
            List of (user_id, fake_ratio) tuples
        """
        suspicious = []

        idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        fake_ratios = self.graph.nodes['user'].data['fake_ratio']

        for user_idx, fake_ratio in enumerate(fake_ratios):
            if fake_ratio >= threshold:
                user_id = idx_to_user[user_idx]
                suspicious.append((user_id, float(fake_ratio)))

        return sorted(suspicious, key=lambda x: x[1], reverse=True)

    def get_suspicious_products(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get products with high fake review ratios

        Args:
            threshold: Minimum fake ratio to consider suspicious

        Returns:
            List of (product_id, fake_ratio) tuples
        """
        suspicious = []

        idx_to_product = {idx: product for product, idx in self.product_to_idx.items()}
        fake_ratios = self.graph.nodes['product'].data['fake_ratio']

        for product_idx, fake_ratio in enumerate(fake_ratios):
            if fake_ratio >= threshold:
                product_id = idx_to_product[product_idx]
                suspicious.append((product_id, float(fake_ratio)))

        return sorted(suspicious, key=lambda x: x[1], reverse=True)

    def save_graph(self, filepath: str):
        """Save graph to file"""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build() first.")

        dgl.save_graphs(filepath, [self.graph])
        print(f"Graph saved to {filepath}")

    def load_graph(self, filepath: str):
        """Load graph from file"""
        graphs, _ = dgl.load_graphs(filepath)
        self.graph = graphs[0]
        print(f"Graph loaded from {filepath}")
        return self.graph


# Testing
if __name__ == "__main__":
    print("Testing HeterographBuilder...")
    print("Note: Requires reviews with user_id and product_id")
    print("For testing, use FraudAmazon dataset or reviews with complete metadata")

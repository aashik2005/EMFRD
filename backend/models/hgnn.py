"""
Heterogeneous Graph Neural Network (HGNN) for Fake Review Detection

Uses graph structure (User-Review-Product relationships) to detect
behavioral patterns indicative of fake reviews.

Architecture:
    Graph: User-Review-Product heterograph
    ↓
    Node Embeddings (learnable for each node type)
    ↓
    Heterogeneous Graph Convolution Layers
    ↓
    Review Node Representations
    ↓
    Classification Head
    ↓
    Fake/Genuine Prediction
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import warnings

try:
    import dgl
    import dgl.nn.pytorch as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not installed. HGNN will not be available.")

from .base import BaseFakeReviewModel


class HGNN(BaseFakeReviewModel):
    """
    Heterogeneous Graph Neural Network for Fake Review Detection

    Args:
        num_users: Number of users in the graph
        num_products: Number of products in the graph
        num_reviews: Number of reviews in the graph
        hidden_dim: Hidden dimension for embeddings (default: 128)
        num_layers: Number of graph convolution layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        num_labels: Number of output classes (default: 2)
    """

    def __init__(
        self,
        num_users: int,
        num_products: int,
        num_reviews: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_labels: int = 2,
    ):
        super().__init__()

        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for HGNN. Install with: pip install dgl")

        self.num_users = num_users
        self.num_products = num_products
        self.num_reviews = num_reviews
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.num_labels = num_labels

        # Node embeddings for each type
        self.user_embed = nn.Embedding(num_users, hidden_dim)
        self.product_embed = nn.Embedding(num_products, hidden_dim)
        self.review_embed = nn.Embedding(num_reviews, hidden_dim)

        # Heterogeneous graph conv layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                dglnn.HeteroGraphConv({
                    'writes': dglnn.GraphConv(hidden_dim, hidden_dim),
                    'about': dglnn.GraphConv(hidden_dim, hidden_dim),
                    'reviews': dglnn.GraphConv(hidden_dim, hidden_dim),
                }, aggregate='mean')
            )

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head for reviews
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.product_embed.weight)
        nn.init.xavier_uniform_(self.review_embed.weight)

        print(f"HGNN initialized:")
        print(f"  Nodes: {num_users} users, {num_products} products, {num_reviews} reviews")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Parameters: {self.count_parameters():,}")

    def forward(
        self,
        graph: dgl.DGLGraph,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            graph: DGL heterograph with node features
            labels: (num_reviews,) - optional for training

        Returns:
            Dictionary containing:
                - logits: (num_reviews, num_labels)
                - embeddings: (num_reviews, hidden_dim) - review embeddings
                - loss: scalar (if labels provided)
        """
        # Get initial embeddings
        h = {
            'user': self.user_embed.weight,
            'product': self.product_embed.weight,
            'review': self.review_embed.weight,
        }

        # Apply graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            # Graph convolution
            h_new = conv(graph, h)

            # Add residual connection and layer norm
            for ntype in h_new:
                h_new[ntype] = self.layer_norms[i](h_new[ntype] + h[ntype])
                h_new[ntype] = self.dropout(torch.relu(h_new[ntype]))

            h = h_new

        # Get review embeddings
        review_embeddings = h['review']

        # Classification
        logits = self.classifier(review_embeddings)

        # Prepare output
        result = {
            "logits": logits,
            "embeddings": review_embeddings,
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": "HGNN",
            "num_users": self.num_users,
            "num_products": self.num_products,
            "num_reviews": self.num_reviews,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_prob,
            "num_labels": self.num_labels,
            "num_parameters": self.count_parameters(),
        }


class HGNNWithFeatures(HGNN):
    """
    HGNN that incorporates node features (ratings, review counts, etc.)

    Extends base HGNN by combining learned embeddings with node features.
    """

    def __init__(
        self,
        num_users: int,
        num_products: int,
        num_reviews: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_labels: int = 2,
        use_node_features: bool = True,
    ):
        super().__init__(
            num_users=num_users,
            num_products=num_products,
            num_reviews=num_reviews,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_labels=num_labels,
        )

        self.use_node_features = use_node_features

        if use_node_features:
            # Feature projection layers
            self.user_feat_proj = nn.Linear(3, hidden_dim)  # review_count, fake_ratio, avg_rating
            self.product_feat_proj = nn.Linear(3, hidden_dim)
            self.review_feat_proj = nn.Linear(1, hidden_dim)  # rating

            nn.init.xavier_uniform_(self.user_feat_proj.weight)
            nn.init.xavier_uniform_(self.product_feat_proj.weight)
            nn.init.xavier_uniform_(self.review_feat_proj.weight)

    def forward(
        self,
        graph: dgl.DGLGraph,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with node features"""

        # Get initial embeddings
        h_embed = {
            'user': self.user_embed.weight,
            'product': self.product_embed.weight,
            'review': self.review_embed.weight,
        }

        # Combine with node features if available
        if self.use_node_features and 'review_count' in graph.nodes['user'].data:
            # User features
            user_feats = torch.stack([
                graph.nodes['user'].data['review_count'],
                graph.nodes['user'].data['fake_ratio'],
                graph.nodes['user'].data['avg_rating'],
            ], dim=1)
            h_embed['user'] = h_embed['user'] + self.user_feat_proj(user_feats)

            # Product features
            product_feats = torch.stack([
                graph.nodes['product'].data['review_count'],
                graph.nodes['product'].data['fake_ratio'],
                graph.nodes['product'].data['avg_rating'],
            ], dim=1)
            h_embed['product'] = h_embed['product'] + self.product_feat_proj(product_feats)

            # Review features
            review_feats = graph.nodes['review'].data['rating'].unsqueeze(1)
            h_embed['review'] = h_embed['review'] + self.review_feat_proj(review_feats)

        h = h_embed

        # Apply graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            h_new = conv(graph, h)

            for ntype in h_new:
                h_new[ntype] = self.layer_norms[i](h_new[ntype] + h[ntype])
                h_new[ntype] = self.dropout(torch.relu(h_new[ntype]))

            h = h_new

        # Get review embeddings
        review_embeddings = h['review']

        # Classification
        logits = self.classifier(review_embeddings)

        # Prepare output
        result = {
            "logits": logits,
            "embeddings": review_embeddings,
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result


# Testing
if __name__ == "__main__":
    print("Testing HGNN...")

    if not DGL_AVAILABLE:
        print("DGL not available. Skipping test.")
    else:
        # Create dummy graph
        num_users = 10
        num_products = 5
        num_reviews = 20

        # Create heterograph
        graph_data = {
            ('user', 'writes', 'review'): (
                torch.tensor([0, 1, 2, 3, 4] * 4),  # user_ids
                torch.tensor(list(range(20)))  # review_ids
            ),
            ('review', 'about', 'product'): (
                torch.tensor(list(range(20))),
                torch.tensor([0, 1, 2, 3, 4] * 4)  # product_ids
            ),
            ('user', 'reviews', 'product'): (
                torch.tensor([0, 1, 2, 3, 4] * 4),
                torch.tensor([0, 1, 2, 3, 4] * 4)
            ),
        }

        graph = dgl.heterograph(graph_data)

        # Add dummy features
        graph.nodes['user'].data['review_count'] = torch.randn(num_users)
        graph.nodes['user'].data['fake_ratio'] = torch.rand(num_users)
        graph.nodes['user'].data['avg_rating'] = torch.randn(num_users)

        graph.nodes['product'].data['review_count'] = torch.randn(num_products)
        graph.nodes['product'].data['fake_ratio'] = torch.rand(num_products)
        graph.nodes['product'].data['avg_rating'] = torch.randn(num_products)

        graph.nodes['review'].data['rating'] = torch.randn(num_reviews)
        graph.nodes['review'].data['label'] = torch.randint(0, 2, (num_reviews,))

        labels = graph.nodes['review'].data['label']

        # Test HGNN
        model = HGNNWithFeatures(
            num_users=num_users,
            num_products=num_products,
            num_reviews=num_reviews,
            hidden_dim=32,
            num_layers=2,
        )

        print(f"\nGraph: {graph}")
        print(f"Labels: {labels}")

        # Forward pass
        outputs = model(graph, labels=labels)

        print(f"\nOutput shapes:")
        print(f"  Logits: {outputs['logits'].shape}")
        print(f"  Embeddings: {outputs['embeddings'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")

        print("\nTest passed!")

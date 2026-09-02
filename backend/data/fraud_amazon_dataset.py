"""
FraudAmazon Dataset Adapter for EMFRD

Uses DGL's FraudAmazon dataset which contains:
- User-Product-Review graph structure
- Fraudulent transaction labels
- Pre-built heterogeneous graph

This dataset is specifically designed for fraud detection with graph neural networks
and is ideal for testing HGNN components when the primary dataset lacks graph metadata.

Dataset Info:
- Source: DGL built-in datasets
- Size: ~11,000 users, ~4,000 products, ~25,000 reviews
- Labels: Fraudulent vs. legitimate transactions
- Graph: Pre-built heterogeneous graph
"""
from typing import List
from pathlib import Path
import warnings

try:
    import dgl
    from dgl.data import FraudAmazonDataset as DGLFraudAmazon
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not installed. FraudAmazon dataset will not be available.")

from .dataset_base import BaseDataset
from .schemas import ReviewRecord, DatasetInfo


class FraudAmazonDataset(BaseDataset):
    """
    Adapter for DGL's FraudAmazon dataset

    This dataset comes pre-built as a heterogeneous graph, making it
    ideal for HGNN experimentation when the primary dataset lacks
    user/product metadata.

    Note: This dataset is for GRAPH experiments only.
    For semantic (RoBERTa) experiments, use the primary fake_reviews dataset.
    """

    def __init__(self, data_dir: Path, cache_dir: Path = None):
        if not DGL_AVAILABLE:
            raise ImportError(
                "DGL is required for FraudAmazon dataset. "
                "Install with: pip install dgl"
            )

        super().__init__(data_dir, cache_dir)
        self.dgl_dataset = None
        self.graph = None

    def download(self) -> None:
        """
        Download FraudAmazon dataset

        This will automatically download from DGL's server
        """
        print("\n" + "="*80)
        print("Downloading FraudAmazon Dataset from DGL")
        print("="*80)
        print("\nThis dataset will be downloaded automatically.")
        print("It contains:")
        print("  - ~11,000 users")
        print("  - ~4,000 products")
        print("  - ~25,000 reviews/transactions")
        print("  - Fraudulent transaction labels")
        print("  - Pre-built heterogeneous graph")
        print("\nDownloading...")

        self.dgl_dataset = DGLFraudAmazon(raw_dir=str(self.data_dir))
        self.graph = self.dgl_dataset[0]

        print(f"\nDownload complete!")
        print(f"Graph: {self.graph}")
        print("="*80 + "\n")

    def load(self):
        """
        Load FraudAmazon dataset

        Returns:
            DGL heterograph (not a DataFrame)
        """
        if self.dgl_dataset is None:
            print("Loading FraudAmazon dataset...")
            try:
                self.dgl_dataset = DGLFraudAmazon(raw_dir=str(self.data_dir))
                self.graph = self.dgl_dataset[0]
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Attempting to download...")
                self.download()

        print(f"\nFraudAmazon Graph loaded:")
        print(f"  Nodes: {self.graph.num_nodes()} total")
        for ntype in self.graph.ntypes:
            print(f"    {ntype}: {self.graph.num_nodes(ntype)}")
        print(f"  Edges: {self.graph.num_edges()} total")
        for etype in self.graph.etypes:
            print(f"    {etype}: {self.graph.num_edges(etype)}")

        return self.graph

    def normalize(self, graph=None) -> List[ReviewRecord]:
        """
        Normalize FraudAmazon to ReviewRecord format

        Note: This dataset doesn't have review text, so we create
        synthetic text based on metadata for compatibility.

        Args:
            graph: DGL graph (ignored, uses self.graph)

        Returns:
            List of ReviewRecord objects
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load() first.")

        print("\nNormalizing FraudAmazon to ReviewRecord format...")
        print("Note: This dataset contains transaction records, not review text.")
        print("Creating synthetic records for compatibility...")

        records = []

        # FraudAmazon has review nodes with labels
        # We'll create synthetic ReviewRecords for compatibility
        num_reviews = self.graph.num_nodes('review') if 'review' in self.graph.ntypes else 0

        if num_reviews == 0:
            # Try other node types that might represent reviews/transactions
            for ntype in self.graph.ntypes:
                if 'label' in self.graph.nodes[ntype].data:
                    num_reviews = self.graph.num_nodes(ntype)
                    review_ntype = ntype
                    print(f"Using node type '{ntype}' as reviews")
                    break
        else:
            review_ntype = 'review'

        if num_reviews == 0:
            raise ValueError("Could not find labeled nodes in FraudAmazon graph")

        # Get labels
        labels = self.graph.nodes[review_ntype].data['label']

        # Create synthetic ReviewRecords
        for idx in range(min(num_reviews, 10000)):  # Limit for memory
            label = int(labels[idx].item())

            # Create synthetic review text based on label
            if label == 1:
                text = f"Synthetic fraudulent transaction record {idx}"
            else:
                text = f"Synthetic legitimate transaction record {idx}"

            record = ReviewRecord(
                review_id=f"fraud_amazon_{idx}",
                review_text=text,
                label=label,
                user_id=f"user_{idx % 1000}",  # Synthetic
                product_id=f"product_{idx % 500}",  # Synthetic
                rating=None,
                timestamp=None,
            )

            records.append(record)

        print(f"Created {len(records)} synthetic ReviewRecords")
        return records

    def validate(self) -> DatasetInfo:
        """
        Validate FraudAmazon dataset

        Returns:
            DatasetInfo with dataset statistics
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load() first.")

        # Get review node type
        review_ntype = 'review' if 'review' in self.graph.ntypes else None
        if review_ntype is None:
            for ntype in self.graph.ntypes:
                if 'label' in self.graph.nodes[ntype].data:
                    review_ntype = ntype
                    break

        if review_ntype is None:
            raise ValueError("Could not find labeled nodes")

        # Get statistics
        labels = self.graph.nodes[review_ntype].data['label']
        total_reviews = len(labels)
        fake_count = int((labels == 1).sum().item())
        genuine_count = int((labels == 0).sum().item())

        # Check for user/product nodes
        has_user = 'user' in self.graph.ntypes or 'U' in self.graph.ntypes
        has_product = 'product' in self.graph.ntypes or 'P' in self.graph.ntypes

        info = DatasetInfo(
            name="FraudAmazon",
            total_reviews=total_reviews,
            fake_count=fake_count,
            genuine_count=genuine_count,
            has_user_id=has_user,
            has_product_id=has_product,
            has_timestamp=False,
            has_rating=False,
            train_size=0,
            val_size=0,
            test_size=0,
            missing_values={},
            duplicate_count=0,
        )

        return info

    def get_graph(self):
        """Get the DGL heterograph directly"""
        if self.graph is None:
            self.load()
        return self.graph

    def prepare(self):
        """
        Prepare dataset - overridden to return graph + info

        For FraudAmazon, we return the graph directly since it's
        more useful than synthetic ReviewRecords.

        Returns:
            Tuple of (graph, dataset_info)
        """
        # Load graph
        graph = self.load()

        # Validate
        info = self.validate()

        print(f"\nFraudAmazon Dataset prepared:")
        print(f"  Total nodes: {graph.num_nodes()}")
        print(f"  Total edges: {graph.num_edges()}")
        print(f"  Fraudulent: {info.fake_count} ({info.fake_count/info.total_reviews*100:.1f}%)")
        print(f"  Legitimate: {info.genuine_count} ({info.genuine_count/info.total_reviews*100:.1f}%)")
        print(f"  Can build graph: {info.can_build_graph}")

        # Return graph instead of records for this dataset
        return graph, info


# Testing
if __name__ == "__main__":
    print("Testing FraudAmazon Dataset Adapter...")

    if not DGL_AVAILABLE:
        print("DGL not available. Skipping test.")
    else:
        from pathlib import Path

        # Create dataset
        data_dir = Path("./data/raw/fraud_amazon")
        dataset = FraudAmazonDataset(data_dir)

        # Prepare
        graph, info = dataset.prepare()

        print("\nDataset Info:")
        for key, value in info.to_dict().items():
            print(f"  {key}: {value}")

        print("\nGraph structure:")
        print(f"  Node types: {graph.ntypes}")
        print(f"  Edge types: {graph.etypes}")

        print("\nTest passed!")

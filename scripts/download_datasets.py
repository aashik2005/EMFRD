"""
Dataset download helper script

Provides instructions and utilities for downloading datasets
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.data import DatasetRegistry, get_dataset
from backend.config import settings


def show_download_instructions():
    """Show download instructions for all datasets"""
    print("="*80)
    print("EMFRD Dataset Download Instructions")
    print("="*80)
    print()

    print("PRIMARY DATASET: Kaggle Fake Reviews")
    print("-" * 80)
    print("Source 1 (Kaggle):")
    print("  URL: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset")
    print("  Steps:")
    print("    1. Log in to Kaggle")
    print("    2. Download the dataset")
    print(f"    3. Extract CSV to: {settings.DATA_DIR / 'raw' / 'fake_reviews'}/")
    print()
    print("Source 2 (OSF):")
    print("  URL: https://osf.io/tyue9/")
    print("  Steps:")
    print("    1. Click 'Download'")
    print(f"    2. Place CSV in: {settings.DATA_DIR / 'raw' / 'fake_reviews'}/")
    print()
    print("Expected filenames:")
    print("  - fake_reviews_dataset.csv")
    print("  - deceptive-opinion.csv")
    print("  - Any .csv file in the directory")
    print()

    print("OPTIONAL DATASETS (for later phases)")
    print("-" * 80)
    print()

    print("1. Amazon Reviews 2023 (large scale evaluation)")
    print("   URL: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023")
    print("   Note: Download specific categories only")
    print()

    print("2. FraudAmazon Dataset (for HGNN)")
    print("   Will be automatically downloaded via DGL")
    print("   Used for graph-based experiments")
    print()

    print("3. Modern Fake Reviews GLM (AI-generated reviews)")
    print("   URL: https://huggingface.co/datasets/Flowerly/modern-fake-reviews-glm")
    print("   Used for robustness testing")
    print()

    print("="*80)
    print("To proceed:")
    print("  1. Download the PRIMARY dataset (Kaggle Fake Reviews)")
    print("  2. Place it in the appropriate directory")
    print("  3. Run: python -m backend.training.train_roberta")
    print("="*80)


def validate_dataset(dataset_name: str):
    """Validate that a dataset can be loaded"""
    print(f"\nValidating dataset: {dataset_name}")
    print("-" * 60)

    try:
        dataset = get_dataset(
            dataset_name,
            data_dir=settings.DATA_DIR / "raw" / dataset_name,
            cache_dir=settings.CACHE_DIR,
        )

        records, info = dataset.prepare()

        print("✓ Dataset loaded successfully!")
        print()
        print("Dataset Information:")
        for key, value in info.to_dict().items():
            print(f"  {key}: {value}")

        print()
        print("Sample reviews:")
        for i, record in enumerate(records[:3]):
            label = "FAKE" if record.label == 1 else "GENUINE"
            text = record.review_text[:100] + "..." if len(record.review_text) > 100 else record.review_text
            print(f"  {i+1}. [{label}] {text}")

        return True

    except FileNotFoundError as e:
        print(f"✗ Dataset not found: {e}")
        print()
        print("Please download the dataset first.")
        return False

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def main():
    """Main function"""
    print("\nEMFRD Dataset Manager\n")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "instructions":
            show_download_instructions()

        elif command == "validate":
            dataset_name = sys.argv[2] if len(sys.argv) > 2 else settings.PRIMARY_DATASET
            validate_dataset(dataset_name)

        elif command == "list":
            print("Available dataset adapters:")
            for name in DatasetRegistry.list_datasets():
                print(f"  - {name}")

        else:
            print(f"Unknown command: {command}")
            print("Available commands: instructions, validate, list")

    else:
        show_download_instructions()


if __name__ == "__main__":
    main()

"""
Create a small demo dataset for testing the training pipeline
This is NOT for research - just to demonstrate the system works
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed
np.random.seed(42)

# Fake review templates (overly positive, repetitive, generic)
fake_templates = [
    "This is absolutely AMAZING!!! Best product ever! 5 stars! Everyone should buy this immediately!",
    "Perfect perfect perfect! I love it so much! Highly recommended! Best purchase ever!!!",
    "Incredible product! Outstanding! Fantastic! Amazing quality! Buy it now!",
    "Best thing I ever bought! Amazing amazing amazing! Five stars! Highly recommend!",
    "Wow wow wow! This is perfect! Love it! Great great great! Must buy!",
    "Excellent product! Super happy! Best ever! Highly satisfied! Amazing quality!",
    "Outstanding! This exceeded all expectations! Perfect! Love love love!",
    "Amazing purchase! Best product! Super satisfied! Highly recommend! Five stars!",
]

# Genuine review templates (balanced, specific, detailed)
genuine_templates = [
    "The product arrived on time. It works as described in the listing. Build quality is decent for the price point.",
    "Purchased this last week. Setup was straightforward. Battery life is about 6 hours with moderate use.",
    "Item matches the description. Shipping took 3 days. Quality is acceptable. Would buy again.",
    "Received the product yesterday. Packaging was secure. Functionality is as expected. Good value.",
    "Used it for a week now. Performance is consistent. A few minor issues but overall satisfied.",
    "The product does what it claims. Installation was easy. Material feels solid. Fair price.",
    "Delivery was prompt. Product quality is reasonable. Met my basic requirements. Decent purchase.",
    "Works fine for everyday use. Not perfect but good enough. Price is competitive. Reasonable quality.",
]

# Generate synthetic dataset
n_samples = 1000  # Small dataset for quick demo
fake_reviews = []
genuine_reviews = []

# Generate fake reviews (with variations)
for i in range(n_samples // 2):
    base = np.random.choice(fake_templates)
    # Add some variation
    variations = ["!!!", "Amazing!", "Perfect!", "Love it!", "Best!"]
    variation = np.random.choice(variations)
    review = f"{base} {variation}"
    fake_reviews.append({
        'text': review,
        'label': 1,  # 1 = fake
        'rating': np.random.choice([4, 5]),  # Fake reviews tend to be 4-5 stars
    })

# Generate genuine reviews (with variations)
for i in range(n_samples // 2):
    base = np.random.choice(genuine_templates)
    # Add some variation
    details = [
        "No major complaints.",
        "Minor issues noted.",
        "As expected.",
        "Satisfactory experience.",
        "Meets requirements.",
    ]
    detail = np.random.choice(details)
    review = f"{base} {detail}"
    genuine_reviews.append({
        'text': review,
        'label': 0,  # 0 = genuine
        'rating': np.random.choice([3, 4, 5]),  # More varied ratings
    })

# Combine and shuffle
all_reviews = fake_reviews + genuine_reviews
np.random.shuffle(all_reviews)

# Create DataFrame
df = pd.DataFrame(all_reviews)

# Save to CSV
output_dir = Path("data/raw/fake_reviews")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "demo_dataset.csv"

df.to_csv(output_file, index=False)

print(f"Created demo dataset: {output_file}")
print(f"Total samples: {len(df)}")
print(f"Fake: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
print(f"Genuine: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
print("\nSample reviews:")
for i, row in df.head(3).iterrows():
    label = "FAKE" if row['label'] == 1 else "GENUINE"
    print(f"{i+1}. [{label}] {row['text'][:80]}...")

print("\n" + "="*60)
print("NOTE: This is a DEMO dataset for testing the pipeline.")
print("For actual research, download the Kaggle Fake Reviews dataset.")
print("="*60)

import csv
import random
from collections import Counter

INPUT_FILE = "train_test_network.csv"
OUTPUT_FILE = "train_test_network_10k.csv"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --------------------------------------------------
# PASS 1: Count labels
# --------------------------------------------------

print("Pass 1: Counting labels...")

label_counts = Counter()

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.DictReader(f)

    if "label" not in reader.fieldnames:
        print("ERROR: Could not find a column named 'label'.")
        print("Columns found:")
        print(reader.fieldnames)
        raise SystemExit

    for row in reader:
        label_counts[row["label"]] += 1

print("\nOriginal label distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count:,}")

print(f"\nTotal rows: {sum(label_counts.values()):,}")


# --------------------------------------------------
# Calculate sample size for each label
# --------------------------------------------------

total_rows = sum(label_counts.values())

target_counts = {}

for label, count in label_counts.items():
    target = round((count / total_rows) * SAMPLE_SIZE)
    target_counts[label] = max(1, target)

# Correct rounding so total = exactly 10,000
difference = SAMPLE_SIZE - sum(target_counts.values())

if difference != 0:
    largest_label = max(label_counts, key=label_counts.get)
    target_counts[largest_label] += difference

print("\nTarget sample distribution:")
for label, count in target_counts.items():
    print(f"  {label}: {count:,}")


# --------------------------------------------------
# PASS 2: Randomly sample rows within each label
# --------------------------------------------------

print("\nPass 2: Sampling rows...")

samples = {label: [] for label in label_counts}
seen = {label: 0 for label in label_counts}

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames

    for row in reader:

        label = row["label"]

        if label not in samples:
            continue

        seen[label] += 1

        target = target_counts[label]

        # Reservoir sampling
        if len(samples[label]) < target:
            samples[label].append(row)
        else:
            position = random.randint(1, seen[label])

            if position <= target:
                samples[label][position - 1] = row


# --------------------------------------------------
# Combine and shuffle
# --------------------------------------------------

sample_rows = []

for label_rows in samples.values():
    sample_rows.extend(label_rows)

random.shuffle(sample_rows)


# --------------------------------------------------
# Save 10K dataset
# --------------------------------------------------

print("\nSaving sample...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(sample_rows)


# --------------------------------------------------
# Verify
# --------------------------------------------------

print("\n====================================")
print("10K DATASET CREATED SUCCESSFULLY")
print("====================================")

print(f"Rows saved: {len(sample_rows):,}")
print(f"Output: {OUTPUT_FILE}")

final_labels = Counter(row["label"] for row in sample_rows)

print("\nFinal label distribution:")

for label, count in final_labels.items():
    print(f"  {label}: {count:,}")
# ACH Transaction Processor Matching

A transaction-description normalization and fuzzy-matching workflow that maps ACH memo text to a set of payment processors, routes low-confidence candidates for review, and summarizes transaction volume by processor and customer.

## Project Files

- `analysis.py`: standalone analysis extracted from the coursework notebook.
- `ach_transactions_assignment.csv`: project input data copied from the coursework folder.
- `report.html` and `report.pdf`: project summary and findings.
- `requirements.txt`: Python dependencies.

## Approach

1. Normalize case and punctuation, remove transaction IDs and banking noise tokens, and expand known processor abbreviations.
2. Compare normalized descriptions with aliases for ten processors using a custom token-set similarity score built on Python's `SequenceMatcher`.
3. Use the notebook's observed score gap to choose a heuristic 80-point confidence threshold. Preserve lower-confidence suggestions in a manual-review queue rather than assigning them as confirmed matches.
4. Aggregate confident matches by processor and customer, including volume, transaction count, average ticket, and processor diversity.

## Recorded Results

The saved notebook output contains 1,000 transactions: 806 are labeled `CONFIDENT_MATCH`, while 194 are flagged for review, representing $400,550.92. The confident-match output totals $3,697,542.35. Toast has the largest recorded confident volume ($415,515.99); the customer-level table ranks accounts by volume and shows processor counts.

## Limitations

- The 80-point cutoff is a heuristic based on an observed score gap, not a threshold validated against labeled ground truth. Precision, recall, and false-match rates are unknown.
- Similarity to a known processor is not proof that a transaction belongs to that processor. The review queue is essential; do not treat this as an autonomous financial decision system.
- The dataset and results are coursework evidence and may be synthetic or otherwise non-production. Inspect the data-sharing terms before publishing the CSV.

## Run

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python analysis.py
```

The script reads the bundled CSV and prints confidence, review-queue, processor, and customer summaries.
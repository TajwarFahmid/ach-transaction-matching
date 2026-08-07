"""ACH transaction processor matching and analysis"""
from pathlib import Path

DATA_PATH = Path(__file__).resolve().with_name("ach_transactions_assignment.csv")

# Extracted from notebook cell 0
import pandas as pd
import numpy as np
import re

# Extracted from notebook cell 1
# Load dataset
df = pd.read_csv(DATA_PATH)


# Extracted from notebook cell 3
# Converting all description strings to lowercase because different financial institutions use different casing, converting all of them to lowercase will ensure every transaction is using the same casing.
df["norm_step1"] = df["ach_description"].astype(str).str.lower()

# Convert delimiters to white spaces because converting these punctuation to white space breaks down compound strings into separate word tokens so merchant names can be identified independently.
df["norm_step2"] = df["norm_step1"].apply(lambda x: re.sub(r"[/\\*_\-.:]", " ", x))

# Stripping numbers and alphanumeric sequence IDs because they don't help in string matching at all.
df["norm_step3"] = df["norm_step2"].apply(lambda x: re.sub(r"\b[a-z0-9]*\d+[a-z0-9]*\b", " ", x))

# Stripping accidental symbols, quotes, commas, or control characters that can appear in unstandardized feeds without corrupting valid alphabetical words.
df["norm_step4"] = df["norm_step3"].apply(lambda x: re.sub(r"[^a-z0-9\s]", " ", x))

# Extracted from notebook cell 4
# Expanding truncated banking codes to ensure downstream fuzzy matching does not miss truncated records.
EXPANSIONS = {
    r"\bsq\b": "square",
    r"\btst\b": "toast",
    r"\bpp\b": "paypal",
    r"\bclvr\b": "clover",
    r"\belvn\b": "elavon",
    r"\bstrpe\b": "stripe",
    r"\bworldpy\b": "worldpay",
    r"\bauthnet\b": "authorize net",
    r"\bauth\s+net\b": "authorize net",
    r"\bfisrv\b": "fiserv",
    r"\bfrst\s*data\b": "first data",
    r"\bfirstdata\b": "first data",
    r"\bfdms\b": "first data",
}

def expand_shorthand(text: str) -> str:
    for pattern, replacement in EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)
    return text

# Expanding abbreviations
df["norm_step5"] = df["norm_step4"].apply(expand_shorthand)

# Extracted from notebook cell 5
# Removing action descriptors isolates the processor entity and prevents false similarity across unrelated entries.
BANKING_NOISE = {
    # Operational indicators
    "ach", "ccd", "ppd", "ctx", "des", "id", "indn", "orig", "trf", "pmt",
    "dir", "dep", "deposit", "merch", "mrch", "mer", "batch", "pos", "fee",
    "credit", "debit", "transfer", "trnsfr", "payment", "payments", "xfer",
    "xfr", "settle", "settlmnt", "stl", "payout", "entry", "dp", "sv", "svc", "svcs",
    # Corporate structure suffixes
    "inc", "llc", "corp", "co", "ltd", "nv"
}

def remove_noise_tokens(text: str) -> str:
    tokens = [
        t for t in text.split()
        if t not in BANKING_NOISE and not t.isdigit() and len(t) > 1
    ]
    return " ".join(tokens)

# Remove noise tokens
df["norm_step6"] = df["norm_step5"].apply(remove_noise_tokens)

# Eliminating irregular interior spacing ensures clean token boundaries for fuzzy string comparison.
# Collapse whitespace and creating the final column
df["normalized_description"] = df["norm_step6"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
df.drop(columns=[f"norm_step{i}" for i in range(1, 7)], inplace=True)

# Extracted from notebook cell 8
from difflib import SequenceMatcher

# Define the canonical target processor list and their recognizable aliases
TARGET_PROCESSORS = {
    "Clover": ["clover"],
    "Square": ["square", "squareup"],
    "Stripe": ["stripe"],
    "Toast": ["toast", "toasttab"],
    "Fiserv / First Data": ["fiserv", "first data"],
    "PayPal": ["paypal"],
    "Worldpay": ["worldpay"],
    "Elavon": ["elavon"],
    "Adyen": ["adyen"],
    "Authorize.Net": ["authorize net", "authorizenet"],
}


# Implement Token Set Ratio fuzzy matching
def token_set_ratio(s1: str, s2: str) -> float:
    t1 = set(s1.split())
    t2 = set(s2.split())

    intersection = t1.intersection(t2)
    diff1 = t1 - intersection
    diff2 = t2 - intersection

    t0_str = " ".join(sorted(intersection))
    t1_str = (t0_str + " " + " ".join(sorted(diff1))).strip()
    t2_str = (t0_str + " " + " ".join(sorted(diff2))).strip()

    ratios = [
        SequenceMatcher(None, t0_str, t1_str).ratio() if t0_str else 0,
        SequenceMatcher(None, t0_str, t2_str).ratio() if t0_str else 0,
        SequenceMatcher(None, t1_str, t2_str).ratio(),
    ]
    return max(ratios) * 100.0


# Matching each normalized description against the target processor list
def match_processor(norm_desc: str):
    if not norm_desc:
        return None, 0.0

    best_target = None
    best_score = 0.0

    for target, aliases in TARGET_PROCESSORS.items():
        for alias in aliases:
            score = token_set_ratio(alias, norm_desc)
            if score > best_score:
                best_score = score
                best_target = target
            elif score == best_score and score == 100.0:
                # Prioritize specific POS brand Clover over backend Fiserv / First Data
                if target == "Clover" and "clover" in norm_desc:
                    best_target = target

    return best_target, round(best_score, 1)


# Apply fuzzy matching across the dataset
match_results = df["normalized_description"].apply(match_processor)
df["matched_processor"] = [r[0] for r in match_results]
df["match_score"] = [r[1] for r in match_results]

# Extracted from notebook cell 12
# sorting unique scores descending
unique_scores = sorted(df["match_score"].unique(), reverse=True)

# extracting the boundary between true matches and noise
min_true_match_score = unique_scores[0]  # 100.0
max_non_merchant_score = unique_scores[1]  # 61.5

# calculating optimal midpoint threshold
evidence_threshold = (min_true_match_score + max_non_merchant_score) / 2
print(f"Optimal Evidence-Based Threshold: {evidence_threshold:.2f}")  # Outputs: 80.75

# Extracted from notebook cell 15
# confidence threshold based on Question 3's evidence gap
CONFIDENCE_THRESHOLD = 80.0

# function for flagging transactions that cannot be confidently matched (< 80.0)
df["flagged_for_review"] = df["match_score"] < CONFIDENCE_THRESHOLD
df["match_status"] = np.where(
    df["flagged_for_review"],
    "MANUAL_REVIEW_FLAGGED",
    "CONFIDENT_MATCH",
)

# preventing forced matches: clear the matched_processor for flagged transactions
df["suggested_processor"] = df["matched_processor"]
df.loc[df["flagged_for_review"], "matched_processor"] = None

# extracting the review queue dataframe for operations/analysts
manual_review_queue = df[df["flagged_for_review"]][
    [
        "customer_id",
        "customer_name",
        "ach_amount",
        "ach_description",
        "normalized_description",
        "suggested_processor",
        "match_score",
    ]
].copy()

# summary metrics
print("--- Transaction Classification Summary ---")
print(df["match_status"].value_counts())
print(f"\nTotal transactions flagged for manual review: {len(manual_review_queue)}")
print(f"Total dollar volume flagged for manual review: ${manual_review_queue['ach_amount'].sum():,.2f}")
print("\nSample transactions in manual review queue:")
print(manual_review_queue.head(10).to_string(index=False))

# Extracted from notebook cell 18
# Filter strictly to confident merchant matches (excluding flagged/non-merchant records)
confident_df = df[df["match_status"] == "CONFIDENT_MATCH"].copy()

# Part A: Aggregate Total Dollar Volume by Processor
processor_summary = (
    confident_df.groupby("matched_processor")
    .agg(
        total_dollar_volume=("ach_amount", "sum"),
        transaction_count=("ach_amount", "count"),
        unique_customers=("customer_id", "nunique"),
        avg_ticket_size=("ach_amount", "mean"),
    )
    .sort_values(by="total_dollar_volume", ascending=False)
    .reset_index()
)

# Format currency columns for clean presentation
processor_summary["total_dollar_volume_fmt"] = processor_summary[
    "total_dollar_volume"
].apply(lambda x: f"${x:,.2f}")
processor_summary["avg_ticket_size_fmt"] = processor_summary[
    "avg_ticket_size"
].apply(lambda x: f"${x:,.2f}")


# Part B: Aggregate Dollar Volume by Customer and Processor
customer_processor_breakdown = (
    confident_df.groupby(["customer_id", "customer_name", "matched_processor"])
    .agg(
        dollar_volume=("ach_amount", "sum"),
        txn_count=("ach_amount", "count"),
    )
    .sort_values(by=["customer_id", "dollar_volume"], ascending=[True, False])
    .reset_index()
)

# Part C: Rank Largest Merchant Services Opportunities by Customer
customer_opportunities = (
    confident_df.groupby(["customer_id", "customer_name"])
    .agg(
        total_merchant_volume=("ach_amount", "sum"),
        transaction_count=("ach_amount", "count"),
        primary_processor=(
            "matched_processor",
            lambda x: x.mode()[0] if not x.empty else "None",
        ),
        unique_processors=("matched_processor", "nunique"),
    )
    .sort_values(by="total_merchant_volume", ascending=False)
    .reset_index()
)

customer_opportunities["total_merchant_volume_fmt"] = customer_opportunities[
    "total_merchant_volume"
].apply(lambda x: f"${x:,.2f}")

# Key Outputs
print("--- Total Dollar Volume by Processor ---")
print(
    processor_summary[
        [
            "matched_processor",
            "total_dollar_volume_fmt",
            "transaction_count",
            "unique_customers",
            "avg_ticket_size_fmt",
        ]
    ].to_string(index=False)
)

print("\n--- Top 10 Customer Merchant Services Opportunities ---")
print(
    customer_opportunities[
        [
            "customer_id",
            "customer_name",
            "total_merchant_volume_fmt",
            "transaction_count",
            "primary_processor",
            "unique_processors",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


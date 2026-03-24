# Semantic Duplicate Detector

> Identify near-duplicate content in large technical documentation corpora using sentence embeddings and cosine similarity.

---

## Motivation

Technical writing teams managing thousands of documentation pages face a silent productivity drain: **semantic duplication**. The same concept gets documented multiple times, in different words, by different authors, in different releases. This content drift:

- increases maintenance costs (every change must be applied in multiple places),
- introduces subtle inconsistencies that confuse end-users,
- makes reuse of approved content fragments unreliable.

Traditional deduplication tools rely on exact or near-exact string matching (diff, fuzzy string matching). They miss paraphrases entirely:

| Fragment A | Fragment B | Identical? | Semantically equivalent? |
|---|---|---|---|
| "Dynamic range compression reduces peak-to-floor difference" | "A compressor narrows the gap between loud and quiet signal levels" | ❌ | ✅ |

This project addresses that gap using **sentence-level semantic embeddings**.

---

## How It Works

```
Raw text fragments
       │
       ▼
┌─────────────────────┐
│   FragmentEmbedder  │  ← sentence-transformers (SBERT)
│  all-MiniLM-L6-v2   │     encodes each fragment into a 384-dim vector
└─────────────────────┘
       │
       ▼
  N × 384 embedding matrix  (L2-normalised)
       │
       ▼
┌─────────────────────┐
│  Pairwise cosine    │  ← O(N²) dot product via NumPy
│  similarity matrix  │
└─────────────────────┘
       │
       ▼
  Threshold filter  →  candidate duplicate pairs
       │
       ▼
  Terminal report  +  CSV export  +  Streamlit dashboard
```

Embeddings are L2-normalised at encode time, so cosine similarity reduces to a dot product — a single matrix multiplication regardless of corpus size.

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/YOUR_USERNAME/semantic-duplicate-detector.git
cd semantic-duplicate-detector
pip install -r requirements.txt
```

**2. Run on sample data (CLI)**

```bash
python main.py
```

**3. Custom threshold**

```bash
python main.py --threshold 0.75
```

**4. Point at your own file** (one fragment per line)

```bash
python main.py --input my_docs.txt --threshold 0.82
```

**5. Launch the interactive dashboard**

```bash
streamlit run app.py
```

---

## Sample Output

```
================================================================================
  SEMANTIC DUPLICATE DETECTOR — Results
================================================================================
  Fragments analysed : 22
  Similarity threshold: 0.80
  Candidate pairs found: 8
================================================================================

╭───────────┬────────────┬──────────────────────────────────────┬──────────────────────────────────────╮
│ Pair      │ Similarity │ Fragment A                           │ Fragment B                           │
├───────────┼────────────┼──────────────────────────────────────┼──────────────────────────────────────┤
│ 3 ↔ 4    │ 0.9201     │ Dynamic range compression reduces... │ A dynamic range compressor lowers... │
│ 12 ↔ 13  │ 0.9118     │ Content should be normalized to...   │ Broadcast specifications require...  │
│ ...       │ ...        │ ...                                  │ ...                                  │
╰───────────┴────────────┴──────────────────────────────────────┴──────────────────────────────────────╯
```

The CSV report is saved to `outputs/duplicates.csv`.

---

## Project Structure

```
semantic-duplicate-detector/
├── main.py               # CLI entry point
├── app.py                # Streamlit dashboard
├── requirements.txt
├── data/
│   └── sample_docs.py    # 22 sample fragments simulating audio/video documentation
├── src/
│   ├── embedder.py       # FragmentEmbedder — sentence-transformers wrapper
│   ├── detector.py       # Pairwise similarity computation + threshold filtering
│   └── reporter.py       # Terminal table + CSV export
└── outputs/              # Generated reports land here
```

---

## CLI Reference

```
usage: main.py [-h] [--input FILE] [--threshold FLOAT] [--model MODEL] [--no-csv] [--output FILE]

options:
  --input FILE        Path to a .txt file (one fragment per line). Default: built-in sample data.
  --threshold FLOAT   Cosine similarity threshold [0.0–1.0]. Default: 0.80
  --model MODEL       sentence-transformers model name. Default: all-MiniLM-L6-v2
  --no-csv            Skip CSV export.
  --output FILE       CSV output path. Default: outputs/duplicates.csv
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **SBERT over TF-IDF** | TF-IDF captures lexical overlap; SBERT captures meaning. Paraphrases with zero word overlap are correctly flagged. |
| **`all-MiniLM-L6-v2`** | 80 MB, 5× faster than `all-mpnet-base-v2`, competitive on STS benchmarks. Configurable — swap without code changes. |
| **L2 normalisation at encode time** | Converts cosine similarity to a dot product, enabling the O(N²) matrix multiply without a per-pair division. |
| **Upper-triangle iteration** | Avoids reporting both (A, B) and (B, A). |
| **Threshold as a parameter** | Different content types have different paraphrase density. 0.80 works well for technical prose; marketing copy may need 0.70. |

---

## Potential Extensions

- **Cluster-level grouping** — use DBSCAN or agglomerative clustering to group entire families of duplicates, not just pairs.
- **Incremental indexing** — maintain a FAISS index for efficient approximate nearest-neighbour search on corpora too large for an N² matrix.
- **Multilingual support** — swap the model for `paraphrase-multilingual-MiniLM-L12-v2` to detect cross-language duplicates in localised documentation.
- **Diff view** — highlight the specific phrases that differ between near-duplicate pairs to accelerate editorial decisions.

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions

---

## Author

Built as a proof-of-concept for NLP-assisted technical documentation management.  
Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

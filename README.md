# 🧾 Multimodal Receipt Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Repository:** [github.com/lukecageSbu/ImageRAGResearch](https://github.com/lukecageSbu/ImageRAGResearch)

A sophisticated AI-powered fraud detection system for receipts and invoices using **section-aware multimodal embeddings** and **LLM-based forensic analysis**.

---

## Overview

This system detects fraudulent receipts by:

1. **Encoding** receipt images into semantic patches using state-of-the-art vision-language models
2. **Segmenting** patches into logical sections (header, items, payment, metadata, footer)
3. **Retrieving** similar legitimate receipts from a vector database
4. **Investigating** suspicious patterns using Claude via AWS Bedrock

The architecture leverages **MaxSim similarity scoring** for accurate retrieval and performs section-by-section fraud analysis to identify tampering, digital alterations, and arithmetic inconsistencies.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Receipt Image  │────▶│  Multimodal Encoder  │────▶│  Vector Store   │
└─────────────────┘     │  (Jina v4/ColQwen2)  │     │    (LanceDB)    │
                        └──────────────────────┘     └────────┬────────┘
                                                              │
                        ┌──────────────────────┐              ▼
                        │  Investigation       │◀──── Similar Cases
                        │  Engine (Claude)     │
                        └──────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │  PASS / FAIL         │
                        │  + Confidence Score  │
                        │  + Detailed Reasoning│
                        └──────────────────────┘
```

---

## Features

- **🔍 Section-Aware Analysis** — Receipts are segmented into 5 semantic sections for targeted fraud detection
- **🧠 Dual Encoder Support** — Choose between Jina Embeddings v4 or ColQwen2.5 for encoding
- **⚡ MaxSim Retrieval** — Multi-vector similarity search for accurate case matching
- **🔬 Forensic Investigation** — LLM-powered analysis detecting:
  - Digital alterations & font mismatches
  - Arithmetic errors (totals, taxes, rounding)
  - Template abuse & placeholder artifacts
  - Merchant identity inconsistencies
- **📊 Comprehensive Evaluation** — Full metrics suite with detailed per-case logging

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS)
- AWS credentials configured for Bedrock access

### Setup

```bash
# Clone the repository
git clone https://github.com/lukecageSbu/ImageRAGResearch.git
cd ImageRAGResearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
torch
transformers
lancedb
boto3
Pillow
tqdm
scikit-learn
colpali-engine
```

---

## Dataset Structure

Organize your receipt images in the following structure:

```
Dataset/
├── train/
│   ├── receipt_001.png
│   ├── receipt_001.txt    # Optional OCR/metadata
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
├── train.txt              # CSV with columns: unique_ID, image, forged
├── val.txt
└── test.txt
```

**Data Format** (train.txt, val.txt, test.txt):
```csv
unique_ID,image,forged
X00016469622,X00016469622.png,0
X00016469623,X00016469623.png,1
```

- `unique_ID`: Unique identifier for each receipt
- `image`: Filename of the receipt image
- `forged`: Label (0 = legitimate, 1 = fraudulent)

---

## Usage

### 1. Training (Build Vector Store)

Index legitimate receipts into the vector store for similarity search:

```bash
python scripts/train.py --model jina
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Encoder model (`jina` or `colqwen`) | `jina` |
| `--threshold` | Similarity threshold for patch filtering | `0.25` |
| `--device` | Device to use (`cuda`, `mps`, `cpu`) | Auto-detect |
| `--vectorstore-path` | Custom path for vector store | `./vectorstore_{model}` |

### 2. Inference (Single Query)

Query a specific receipt against the indexed database:

```bash
python scripts/inference.py --model jina --index 0
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Encoder model to use | `jina` |
| `--index` | Test set index to query | `0` |
| `--top-k` | Total results to return | `15` |
| `--k-per-section` | Results per section | `3` |
| `--sections` | Specific sections to search | All sections |

### 3. Evaluation

Run comprehensive evaluation on test or validation set:

```bash
python scripts/evaluate.py --model jina --eval-type test
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Encoder model | `jina` |
| `--eval-type` | Dataset split (`test` or `validation`) | `test` |
| `--k-similar` | Similar cases to retrieve | `15` |
| `--top-k` | Top K results per section | `5` |
| `--no-detailed-logs` | Disable per-case logging | Enabled |

**Output:**
- Console metrics (Accuracy, Precision, Recall, F1)
- Session logs in `./evaluation_logs/`
- Detailed per-case reports
- Results CSV file

---

## Configuration

All settings are managed in `configs/default.py`:

```python
@dataclass
class FraudDetectionConfig:
    # Paths
    BASE_DIR: str = "/path/to/Dataset"
    VECTOR_STORE_PATH: str = "./vectorstore"

    # Models
    MODEL_NAME: str = "jinaai/jina-embeddings-v4"
    COLQWEN_MODEL_NAME: str = "vidore/colqwen2-v0.1"
    INVESTIGATOR_MODEL: str = "global.anthropic.claude-opus-4-5-20251101-v1:0"

    # Embedding Settings
    SIMILARITY_THRESHOLD: float = 0.25
    EMBEDDING_DIM: int = 128
    MAX_PATCHES: int = 200

    # Retrieval Settings
    DEFAULT_TOP_K: int = 15
    DEFAULT_TOP_K_PER_SECTION: int = 3

    # AWS
    AWS_REGION: str = "us-east-1"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BASE_DIR` | Override default dataset path |
| `AWS_ACCESS_KEY_ID` | AWS credentials for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for Bedrock |

---

## Receipt Sections

The encoder segments receipts into 5 semantic sections:

| Section | Content |
|---------|---------|
| **Header** | Merchant name, logo, address, phone, registration numbers |
| **Items** | Product list, quantities, unit prices, line totals, SKUs |
| **Payment** | Subtotal, taxes (GST/SST/VAT), service charges, grand total, payment method |
| **Metadata** | Date, time, receipt number, cashier ID, terminal number |
| **Footer** | Thank you message, return policy, barcodes, QR codes |

---

## Fraud Detection Criteria

The investigation engine analyzes:

### 1. Digital Alteration & Template Abuse
- Font mismatches within sections
- Text misalignment with column grids
- Pixelation artifacts around key numbers
- Template placeholder text

### 2. Arithmetic & Logic Validation
- Item prices → Subtotal calculation
- Tax calculation accuracy
- Rounding adjustment logic
- Cash tendered − Change = Grand Total

### 3. Merchant Identity & Layout
- Logo distortion or quality issues
- Address/phone typos
- Layout consistency with similar merchants

### 4. Content Plausibility
- Logical transaction times
- Items consistent with merchant type

---

## Project Structure

```
ImageRAGResearch/
├── configs/
│   └── default.py          # Configuration dataclass
├── scripts/
│   ├── train.py            # Build vector store
│   ├── inference.py        # Single query inference
│   └── evaluate.py         # Full evaluation suite
├── src/
│   ├── data/
│   │   └── dataset.py      # Data loading utilities
│   ├── detection/
│   │   └── fraud_detector.py   # LLM investigation engine
│   ├── models/
│   │   └── multimodal_encoder.py   # Jina/ColQwen encoder
│   └── retrieval/
│       └── vector_store.py     # LanceDB vector store
├── Dataset/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── *.txt               # Data manifests
├── requirements.txt
└── README.md
```

---

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| `jinaai/jina-embeddings-v4` | Encoder | Multi-vector text-image embeddings |
| `vidore/colqwen2-v0.1` | Encoder | ColPali-style document embeddings |
| `claude-opus-4-5-20251101-v1:0` | Investigator | Forensic analysis via AWS Bedrock |

---

## Example Output

```
================================================================================
EVALUATION COMPLETE
================================================================================
Accuracy: 0.8750
Precision: 0.8421
Recall: 0.9143
F1-Score: 0.8767
================================================================================

Confusion Matrix:
[[45  7]
 [ 3 32]]

Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.87      0.90        52
           1       0.82      0.91      0.87        35
    accuracy                           0.89        87
================================================================================
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Jina AI](https://jina.ai/) for multimodal embeddings
- [ColPali](https://github.com/illuin-tech/colpali) for document understanding
- [LanceDB](https://lancedb.com/) for vector storage
- [Anthropic Claude](https://anthropic.com/) via AWS Bedrock for forensic analysis

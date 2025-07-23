# B5W4: Amharic E-commerce Data Extractor for FinTech Analysis

An NLP system to extract structured data from Amharic e-commerce posts on Telegram and generate a "Vendor Scorecard" for FinTech analysis.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Hugging%20Face%20Transformers-yellow.svg)
![Library](https://img.shields.io/badge/Library-Telethon-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 1. Overview

EthioMart's vision is to become the primary hub for Telegram-based e-commerce in Ethiopia. Currently, the market is fragmented across numerous independent channels, creating a decentralized and inefficient experience for both vendors and customers. This project is the foundational phase in developing a centralized platform to solve this issue.

The primary objective is to build a robust Amharic Named Entity Recognition (NER) system that can automatically ingest, process, and extract structured business information from unstructured Telegram posts. This structured data is the key to creating a "Vendor Scorecard," a FinTech tool that will help EthioMart identify promising and reliable vendors for services like micro-lending and logistics partnerships.

### Key Features

- **Automated Data Ingestion:** Programmatically scrapes thousands of posts from public Amharic e-commerce Telegram channels.
- **Named Entity Recognition (NER):** Identifies and extracts key business entities like PRODUCT, PRICE, and LOCATION.
- **High-Quality Labeled Data:** Provides a manually annotated dataset in the standard CoNLL format, ready for model training.
- **Vendor Analytics Foundation:** The extracted data powers a "Vendor Scorecard" by analyzing product types, pricing strategies, post frequency, and market reach (via post views).

---

## 2. Technology Stack

- **Data Ingestion:** Python, Telethon
- **Data Processing:** Pandas, NumPy
- **NLP Model:** Hugging Face Transformers (for fine-tuning models like XLM-Roberta, mBERT)
- **Data Annotation:** CoNLL (BIO) format
- **Development Environment:** Jupyter Notebooks, Python Virtual Environments

---

## 3. Data Workflow & NER Model

The project follows a multi-stage data pipeline to transform raw, unstructured text into valuable, structured insights.

### 3.1. Data Ingestion & Preprocessing

A Python script using the Telethon library connects to the Telegram API to collect data from a curated list of over 20 popular Ethiopian e-commerce channels.

- **Process:** The script fetches the 500 most recent posts from each target channel, extracting post text, message ID, timestamp, and view count.
- **Result:** Over 6,000 text-based posts were collected and aggregated into a single .csv file. The view count metadata is a critical proxy for a post's market reach.

| Channel Username      | Messages Collected |
| --------------------- | ------------------ |
| @ZemenExpress         | 500                |
| @nevacomputer         | 488                |
| @helloomarketethiopia | 500                |
| @Shewabrand           | 451                |
| @meneshayeofficial    | 495                |

### 3.2. Data Labeling for Named Entity Recognition

To train a model to automatically extract information, we manually annotated a representative subset of over 50 posts using the industry-standard CoNLL format with the BIO (Beginning, Inside, Outside) tagging scheme.

#### Entity Schema

The following key entities were defined for extraction:

- **B-PRODUCT / I-PRODUCT:** The beginning and inside of a product name (e.g., "HP Envy Laptop").
- **B-PRICE / I-PRICE:** The beginning and inside of a price mention (e.g., "95000 birr").
- **B-LOC / I-LOC:** The beginning and inside of a location (e.g., "መገናኛ ማራቶን").
- **O:** (Outside) Any token that does not belong to the entities above.

#### Labeling Example

**Raw Telegram Post:**

```
︎︎︎︎︎ : HP ENVY X360 ❤️ Price 95000 birr 🏢 አድራሻ: - መገናኛ ማራቶን የ ገበያ ማእከል
```

**CoNLL Labeled Version (amharic_ner_train.txt):**

```
︎︎︎︎︎ O
: O
HP B-PRODUCT
ENVY I-PRODUCT
X360 I-PRODUCT
❤️ O
Price O
95000 B-PRICE
birr I-PRICE
🏢 O
አድራሻ: O
- O
መገናኛ B-LOC
ማራቶን I-LOC
የ I-LOC
ገበያ I-LOC
ማእከል I-LOC
```

---

## 4. Project Structure

```
.
├── data
│   ├── raw/scraped_telegram_data.csv    # Raw data from Telegram
│   └── processed/amharic_ner_train.txt  # Labeled data in CoNLL format
├── notebooks
│   ├── 01_data_ingestion.ipynb          # Notebook for scraping data
│   └── 02_model_finetuning.ipynb        # Notebook for training the NER model
├── scripts
│   ├── ingest_data.py                   # Standalone script for data collection
│   └── extractor.py                     # Script to run inference with the trained model
├── requirements.txt                     # Project dependencies
└── README.md                            # This file
```

---

## 5. Setup and Installation

### Clone the repository

```bash
git clone https://github.com/your-username/amharic-fintech-extractor.git
cd amharic-fintech-extractor
```

### Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install the required dependencies

```bash
pip install -r requirements.txt
```

### Set up Telegram API Credentials

- You need to get your own `api_id` and `api_hash` from [my.telegram.org](https://my.telegram.org).
- Create a file named `.env` in the root directory.
- Add your credentials to the `.env` file like this:

```
API_ID=1234567
API_HASH=0123456789abcdef0123456789abcdef
```

---

## 6. How to Run

### a) Ingest Data from Telegram

To collect fresh data from the target channels, run the ingestion script. The output will be saved in `data/raw/`.

```bash
python scripts/ingest_data.py
```

(Alternatively, you can run the `notebooks/01_data_ingestion.ipynb` notebook cell by cell.)

### b) Train the NER Model

The core model training and fine-tuning logic is located in the Jupyter Notebook. Open and run the cells to train the model on the labeled data.

```bash
jupyter notebook notebooks/02_model_finetuning.ipynb
```

This notebook will guide you through loading the data, setting up the tokenizer and model, and launching the training process. The final trained model will be saved to a local directory.

### c) Run Inference on New Text

Once a model is trained, you can use the `extractor.py` script to extract entities from any Amharic text.

```bash
python scripts/extractor.py "HP ENVY X360 ❤️ Price 95000 birr 🏢 አድራሻ: - መገናኛ ማራቶን"
```

**Expected Output:**

```json
{
  "text": "HP ENVY X360 ❤️ Price 95000 birr 🏢 አድራሻ: - መገናኛ ማራቶን",
  "entities": [
    { "entity_group": "PRODUCT", "word": "HP ENVY X360", "score": 0.98 },
    { "entity_group": "PRICE", "word": "95000 birr", "score": 0.99 },
    { "entity_group": "LOC", "word": "መገናኛ ማራቶን", "score": 0.95 }
  ]
}
```

---

## 7. Next Steps

The successful completion of data ingestion and labeling has laid the foundation for the model development phase. The immediate next steps are:

- **Model Fine-tuning:** Use the high-quality labeled data to fine-tune and compare several state-of-the-art transformer models, including XLM-Roberta and mBERT.
- **Performance Evaluation:** Identify the best-performing model by measuring accuracy, precision, recall, and F1-score on a held-out test set.
- **Vendor Scorecard Development:** Begin designing the logic for the FinTech Vendor Scorecard, using the extracted structured data as input.
- **Expand Entity Schema:** Explore adding more entities like CONTACT_INFO (phone numbers) and CONDITION (new, used) to enrich the data.

---

## 8. Contributing

Contributions are welcome! Please feel free to open an issue to report bugs, suggest features, or submit a pull request.

---

## 9. License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## 10. Author

**Miheret Girmachew**

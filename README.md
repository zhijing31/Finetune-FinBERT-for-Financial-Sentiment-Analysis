# 🧠 FinBERT Sentiment Analysis

This repository contains code and experiments for **Financial Sentiment Analysis (FSA)** using **FinBERT** and **Large Language Models (LLMs)**.  
The project explores how domain-specific fine-tuning improves sentiment classification performance on financial text data.

---

## 📘 1. Project Overview

This notebook compares the performance of:
- **FinBERT** (pretrained on financial corpora)
- **Large Language Models (LLMs)** under zero-shot, few-shot, and fine-tuned settings
- **PEFT / LoRA** lightweight fine-tuning vs **Full Fine-tuning**

We evaluate performance using standard metrics:
- Accuracy
- Precision
- Recall
- F1-score

This study highlights the effectiveness of **domain-specific pretraining** and **prompt engineering** techniques for financial sentiment analysis.

---

## ⚙️ 2. Installation & Environment Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/finbert-sentiment-analysis.git
cd finbert-sentiment-analysis
````

### Step 2 — Install dependencies

You can install all required packages using:

```bash
pip install -r requirements.txt
```

If you prefer manual installation:

```bash
pip install transformers datasets torch scikit-learn pandas numpy peft
```

### Step 3 — Run the notebook

Open Jupyter Notebook or VSCode and execute:

```bash
jupyter notebook finbert-sentiment-analysis.ipynb
```

This notebook will:

* Load the Financial PhraseBank dataset
* Fine-tune FinBERT (full + LoRA versions)
* Evaluate and compare both models

---

## 📂 3. File Structure

```
finbert-sentiment-analysis/
├── finbert-sentiment-analysis.ipynb   # Main notebook
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 📊 4. Dataset

We use the **Financial PhraseBank** dataset available on Kaggle:

📦 [https://www.kaggle.com/datasets/jannesklaas/financial-phrasebank]([https://www.kaggle.com/datasets/jannesklaas/financial-phrasebank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data))

Please **do not upload large datasets or pretrained models**.
To reproduce results, download the dataset manually or from Kaggle and place it under:

```
data/
└── FinancialPhraseBank.csv
```

If you use Kaggle API:

```bash
kaggle datasets download -d jannesklaas/financial-phrasebank
unzip financial-phrasebank.zip -d data/
```

---

## 🧪 5. Reproducing Experiments

1. Open the notebook `finbert-sentiment-analysis.ipynb`
2. Run all cells sequentially

   * Section 1: Environment setup
   * Section 2: Dataset loading
   * Section 3: Full fine-tuning
   * Section 4: LoRA / PEFT fine-tuning
   * Section 5: Evaluation metrics
3. Compare the metrics printed at the end of each experiment (accuracy, precision, recall, F1)

---

## 🧾 6. Dependencies

All dependencies are listed in `requirements.txt`.
Main libraries:

* `transformers` — FinBERT and LLM model loading
* `datasets` — for handling dataset splits
* `torch` — model training and fine-tuning
* `scikit-learn` — evaluation metrics
* `pandas`, `numpy` — data handling
* `peft` — LoRA parameter-efficient fine-tuning

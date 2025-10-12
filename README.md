# Finetune FinBERT with LoRA for Financial Sentiment Analysis

This repository provides code to reproduce fine-tuning experiments using **FinBERT** on the **Financial PhraseBank** dataset.  
It includes both **full fine-tuning** and **LoRA (PEFT)** versions for comparison.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/zhijing31/Finetune-FinBERT-with-LoRA-for-financial-sentiment-analysis.git
cd Finetune-FinBERT-with-LoRA-for-financial-sentiment-analysis
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset used is the **Financial PhraseBank** from Kaggle.
Please download it manually:

🔗 [https://www.kaggle.com/datasets/jannesklaas/financial-phrasebank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data)

After downloading, place the file in:

```
data/
└── FinancialPhraseBank.csv
```

---

## 🧠 Pretrained Model

We use the publicly available **FinBERT** model released by **ProsusAI** on Hugging Face:

🔗 [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

During fine-tuning, the script will automatically download the model from Hugging Face when running:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
```

---

## 🚀 Run Experiments

To reproduce the experiments, open the Jupyter Notebook:

```bash
jupyter notebook finbert-sentiment-analysis.ipynb
```

Then:

1. Run all cells in order.
2. The notebook will:

   * Load the Financial PhraseBank dataset
   * Fine-tune FinBERT using both **full fine-tuning** and **LoRA** methods
   * Display metrics (Accuracy, Precision, Recall, F1-score)


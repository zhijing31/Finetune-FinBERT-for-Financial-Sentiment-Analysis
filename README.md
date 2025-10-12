#  Finetune FinBERT for Financial Sentiment Analysis

This repository contains the Kaggle notebook used to fine-tune **FinBERT** on the **Financial PhraseBank** dataset, comparing **Full Fine-tuning** and **LoRA (PEFT)** methods for financial sentiment classification.

---

### Repository Structure

```

Finetune-FinBERT-with-LoRA-for-financial-sentiment-analysis/
├── Finetune FinBERT                                            # Kaggle notebook
└── README.md                                                   # Project documentation
└── finetune-finbert-for-financial-sentiment-analysis.ipynb     # Code file
```

---
### Dataset

The dataset used in this project is the **Financial PhraseBank** from Kaggle.

🔗 [financialphrasebank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data)

It will be automatically loaded when running the Kaggle notebook — no manual download is required.

---

### Pretrained Model

The pretrained model used is **FinBERT**, available on Hugging Face.

🔗 [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

The model will be automatically downloaded when the notebook is executed.

---

### How to Reproduce

All experiments can be reproduced directly on Kaggle using the notebook below:

🔗 [**Open Kaggle Notebook**](https://www.kaggle.com/code/chenzhijing3121/finbert)

Steps:
1. Open the notebook on Kaggle.  
2. The notebook will:  
   - Load the **Financial PhraseBank** dataset  
   - Fine-tune **FinBERT** using both **Full Fine-tuning** and **LoRA (PEFT)**  
   - Display and compare metrics (Accuracy, Precision, Recall, F1-score)





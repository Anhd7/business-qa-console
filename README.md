# 📈 Business Intelligence Q&A Console

An interactive AI-powered tool that lets you ask natural language questions about financial data (from a CSV) and get **instant answers or predictions** — via Web UI or CLI.

---

## 🧠 What It Does

Ask questions like:

- “What is **Faizan Ali Khan’s** revenue in **Q3**?”
- “Did **Ashish’s** revenue increase from **Q2 to Q3**?”
- “Forecast **Kaustubh’s** revenue for next year Q1.”

---

## 🧰 Features

✅ Fine-tuned **BERT QA model**  
✅ **ML predictions** via Linear Regression & Random Forest  
✅ Intelligent **alias & quarter mapping**  
✅ Choose between **Streamlit UI** or **Command-line interface**  
✅ **CSV-driven** — just update the data sheet

---

## ⚙️ Tech Stack

| Component           | Technology                         |
|--------------------|------------------------------------|
| QA Model           | HuggingFace Transformers (BERT)    |
| Frontend           | Streamlit                          |
| Forecasting Models | Scikit-learn (Linear & RF)         |
| Data Parsing       | pandas + regex + fuzzywuzzy        |

---

## 🚀 Quick Start

### 1. Clone and Set Up

```bash
git clone https://github.com/Anhd7/qa-financial-console.git
cd qa-financial-console

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Train QA Model (Run Once)

```bash
python dataset_builder.py
python train_model.py
```

---

## 🖥️ Web Interface (Streamlit)

```bash
streamlit run app.py
```

- Real-time UI with model status indicators
- Auto-formatted answers with Indian currency

---

## 💻 Command Line Interface

```bash
python main.py
```

You'll see:
```
CSV Q&A System — type 'exit' to quit.
❓ Your question: What was the revenue of faizan in Q2?
🧠 Faizan Ali Khan in Q2 is 12000.
```

---

## 🧾 Dataset Format

Your CSV file should look like:

| Business Head     | Q1   | Q2   | Q3   | Q4   |
|-------------------|------|------|------|------|
| Faizan Ali Khan   | 1000 | 1200 | 1500 | 1800 |
| Ashish            | 900  | 950  | 970  | 1000 |

📝 Aliases like `ashish`, `faizan`, `kaustubh` are automatically mapped.

---

## 🔮 Smart Capabilities

- "**next year Q1**" is auto-understood as Q5
- Aliases like "ashish" → "Ashish" (case-insensitive)
- Computes **yes/no answers**, **percentage growth**, or **forecasted revenue**
- Uses fallback model (Average Growth) if data is sparse

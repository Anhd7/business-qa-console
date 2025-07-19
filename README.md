```markdown
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

---

## ✅ Run Tests with `test.py`

To verify that everything is working, run the included test suite:

```bash
pytest test.py -v
```

This will:
- Check basic Q&A queries
- Validate predictions
- Handle edge cases like unknown topics or missing quarters

> ⚠️ Make sure the model (`qa_finetuned/`) and `Business Heads.csv` are in place, or the tests may fail.

You can add your own test cases in `test.py` to extend or customize this behavior for your data.

---

## 🔄 Use Your Own CSV File

Want to use your own quarterly financial data?

### 1. Prepare Your CSV

Structure your file like this:

| Business Head     | Q1   | Q2   | Q3   | Q4   |
|-------------------|------|------|------|------|
| John Doe          | 1200 | 1500 | 1800 | 2100 |
| Jane Smith        | 1000 | 1100 | 1050 | 1300 |

- You can rename `Business Head` to something else (like `Department`, `Region`, or `Manager`)
- Make sure quarter columns follow the format `Q1`, `Q2`, `Q3`, `Q4`

### 2. Update `config.py`

```python
MODEL_CONFIG = {
    "model_path": "qa_finetuned",
    "csv_path": "YourNewFile.csv",  # <-- Update this
    "topic_column": "YourColumnName"  # <-- Match your CSV column header
}
```

### 3. Rebuild and Retrain

```bash
python dataset_builder.py
python train_model.py
```

Now you can ask questions like:  
> *"What is the revenue of Jane Smith in Q2?"*  
> *"Did John Doe’s revenue increase from Q1 to Q3?"*

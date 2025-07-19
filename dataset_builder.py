import pandas as pd
import json
from config import MODEL_CONFIG


def load_csv(file_path=None):
    file_path = file_path or MODEL_CONFIG["csv_path"]
    df = pd.read_csv(file_path)
    topic_col = MODEL_CONFIG["topic_column"]
    df[topic_col] = df[topic_col].astype(str).str.strip().str.lower()
    return df


def generate_basic_examples(df, topic_col):
    examples = []
    for _, row in df.iterrows():
        topic = row[topic_col]
        for col in df.columns:
            if col.lower().startswith("q") and pd.notna(row[col]):
                question = f"What is the value of {topic} in {col.upper()}?"
                context = f"The value of {topic} in {col.upper()} is {row[col]}."
                answer_text = str(row[col])
                start = context.find(answer_text)
                if start != -1:
                    examples.append({
                        "context": context,
                        "question": question,
                        "answers": [{"text": answer_text, "answer_start": start}]
                    })
    return examples


def generate_comparison_examples(df, topic_col):
    examples = []
    for _, row in df.iterrows():
        topic = row[topic_col]
        values = {col.lower(): row[col] for col in df.columns if col.lower().startswith("q")}
        qtrs = list(values.keys())
        for i in range(len(qtrs) - 1):
            q1, q2 = qtrs[i], qtrs[i + 1]
            val1, val2 = values[q1], values[q2]
            if pd.notna(val1) and pd.notna(val2):
                try:
                    val1 = float(str(val1).replace(",", ""))
                    val2 = float(str(val2).replace(",", ""))
                    change = (val2 - val1) / val1 * 100
                    label = "yes" if val2 > val1 else "no"
                    pct = f"{abs(change):.2f}%"

                    context = (
                        f"The {topic} {'increased' if val2 > val1 else 'decreased'} "
                        f"from {val1} in {q1.upper()} to {val2} in {q2.upper()} ({pct}). The answer is {label}."
                    )

                    start = context.find(label)
                    if start != -1:
                        examples.append({
                            "context": context,
                            "question": f"Did {topic} increase from {q1.upper()} to {q2.upper()}?",
                            "answers": [{"text": label, "answer_start": start}]
                        })

                    start_pct = context.find(pct)
                    if start_pct != -1:
                        examples.append({
                            "context": context,
                            "question": f"What was the percentage change in {topic} from {q1.upper()} to {q2.upper()}?",
                            "answers": [{"text": pct, "answer_start": start_pct}]
                        })

                except:
                    continue
    return examples


def build_dataset():
    df = load_csv()
    topic_col = MODEL_CONFIG["topic_column"]
    basic_examples = generate_basic_examples(df, topic_col)
    comparison_examples = generate_comparison_examples(df, topic_col)
    all_examples = basic_examples + comparison_examples

    squad_data = {
        "data": [{
            "title": "financial_dataset",
            "paragraphs": [
                {
                    "context": ex["context"],
                    "qas": [{
                        "id": f"q_{i}",
                        "question": ex["question"],
                        "answers": ex["answers"],
                        "is_impossible": False
                    }]
                }
                for i, ex in enumerate(all_examples)
            ]
        }]
    }

    with open("squad.json", "w") as f:
        json.dump(squad_data, f, indent=2)


if __name__ == "__main__":
    build_dataset()

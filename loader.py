import pandas as pd
from config import MODEL_CONFIG 

def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]  

    topic_col = MODEL_CONFIG.get("topic_column", "Business Head")
    if topic_col in df.columns:
        df[topic_col] = df[topic_col].astype(str).str.strip().str.lower()
    else:
        raise KeyError(f"❌ Column '{topic_col}' not found in CSV. Available columns: {df.columns.tolist()}")

    return df

def chunk_csv_as_text(df, chunk_size=800):
    text = df.to_string(index=False)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

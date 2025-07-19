
import re
from fuzzywuzzy import process
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from config import TOPIC_ALIASES, QUARTER_MAPPING, MODEL_CONFIG
from loader import load_csv
from ml_predictor import train_all_models_and_rank
import pandas as pd

def convert_natural_quarter_phrasing(question: str, current_max_quarter: int = 4) -> str:
    original_query = question.lower()

    match = re.search(r"(q[1-4])(?:\s+of)?\s+(next year|coming year|following year)", original_query)
    if match:
        q = match.group(1)
        q_num = int(q[1])
        new_q = (current_max_quarter - (current_max_quarter % 4)) + q_num + 4
        question = re.sub(re.escape(match.group(0)), f"Q{new_q}", question, flags=re.IGNORECASE)

    match = re.search(r"(next|second|third|fourth)\s+quarter", original_query)
    if match:
        mapping = {"next": 1, "second": 2, "third": 3, "fourth": 4}
        offset = mapping.get(match.group(1))
        new_q = current_max_quarter + offset
        question = re.sub(re.escape(match.group(0)), f"Q{new_q}", question, flags=re.IGNORECASE)

    match = re.search(r"in\s+(\d+)\s+quarters?", original_query)
    if match:
        offset = int(match.group(1))
        new_q = current_max_quarter + offset
        question = re.sub(re.escape(match.group(0)), f"Q{new_q}", question, flags=re.IGNORECASE)

    return question


def format_currency(number):
    s = str(int(float(number)))
    r = s[-3:]
    s = s[:-3]
    while len(s) > 2:
        r = s[-2:] + "," + r
        s = s[:-2]
    if s:
        r = s + "," + r
    return f"Rs {r}"

class FinancialQASystem:
    def __init__(self, model_path=None, csv_path=None):
        self.model_path = model_path or MODEL_CONFIG["model_path"]
        self.csv_path = csv_path or MODEL_CONFIG["csv_path"]
        self.topic_column = MODEL_CONFIG["topic_column"].lower()
        self.df = load_csv(self.csv_path)
        self.df.columns = [col.lower() for col in self.df.columns]
        self.df[self.topic_column] = self.df[self.topic_column].str.lower()
        self.qa_pipeline = self._load_model()
        self.models, self.best_model_map = train_all_models_and_rank(self.df)

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
        return pipeline("question-answering", model=model, tokenizer=tokenizer)

    def parse_question(self, question):
        question = question.lower()
        quarter = next((mapped for term, mapped in QUARTER_MAPPING.items() if term in question), None)
        topic = next((canonical for alias, canonical in TOPIC_ALIASES.items() if alias in question), None)

        if not topic:
            best_match, score = process.extractOne(
                question, self.df[self.topic_column].tolist()
            )
            topic = best_match.strip().lower() if score > 60 else None

        return topic, quarter

    def get_value(self, topic, quarter):
        topic = topic.strip().lower()
        quarter = quarter.strip().lower() if quarter else None

        if topic not in self.df[self.topic_column].values:
            return None

        row = self.df[self.df[self.topic_column] == topic]
        if quarter in self.df.columns:
            value = row[quarter].values[0]
            return value if pd.notna(value) else None

        return None

    def handle_complex_query(self, question):
        question_lower = question.lower()
        topic, _ = self.parse_question(question_lower)
        if not topic:
            return "❌ Could not identify the topic."

        if any(word in question_lower for word in ["predict", "forecast", "estimate", "next quarter", "future"]):
            return self.predict_future(topic, question_lower)

        if "did" in question_lower or "was" in question_lower:
            return self._process_yesno(question_lower, topic)
        elif "growth" in question_lower or "change" in question_lower:
            return self._process_growth(question_lower, topic)
        elif "compare" in question_lower or "vs" in question_lower:
            return self._process_comparison(question_lower, topic)
        return None

    def _extract_quarters(self, question):
        matches = []
        for phrase, mapped in QUARTER_MAPPING.items():
            if phrase in question:
                matches.append(mapped)
        seen = set()
        unique_matches = []
        for m in matches:
            if m not in seen:
                unique_matches.append(m)
                seen.add(m)
        return unique_matches



    def _process_yesno(self, question, topic):
        qtrs = self._extract_quarters(question)
        if len(qtrs) >= 2:
            val1, val2 = self.get_value(topic, qtrs[0]), self.get_value(topic, qtrs[1])
            if val1 and val2:
                val1, val2 = float(val1), float(val2)
                increase = val2 > val1
                direction = "increased" if increase else "decreased"
                change = (val2 - val1) / val1 * 100
                answer = "yes" if (
                    ("increase" in question and increase) or 
                    ("decrease" in question and not increase)
                ) else "no"
                return f"{topic} {direction} from {val1} to {val2} ({abs(change):.2f}%). Answer: {answer}"
        return f"❌ Could not find {topic} data for required quarters."

    def _process_growth(self, question, topic):
        qtrs = self._extract_quarters(question)
        if "past year" in question or "year over year" in question:
            qtrs = ["q3", "q4"]
        if len(qtrs) >= 2:
            val1, val2 = self.get_value(topic, qtrs[0]), self.get_value(topic, qtrs[1])
            if val1 and val2:
                val1, val2 = float(val1), float(val2)
                change = (val2 - val1) / val1 * 100
                direction = "increased" if change > 0 else "decreased"
                return f"{topic} {direction} by {abs(change):.2f}% from {qtrs[0]} to {qtrs[1]}"
        return f"❌ Could not compute growth for {topic}."

    def _process_comparison(self, question, topic):
        qtrs = self._extract_quarters(question)
        if len(qtrs) >= 2:
            val1 = self.get_value(topic, qtrs[0])
            val2 = self.get_value(topic, qtrs[1])

            print(f"[DEBUG] {topic} - {qtrs[0]}: {val1}, {qtrs[1]}: {val2}")

            try:
                val1 = float(val1)
                val2 = float(val2)
                change = (val2 - val1) / val1 * 100
                direction = "Increase" if change >= 0 else "Decrease"
                return f"{topic} changed from {val1} ({qtrs[0]}) to {val2} ({qtrs[1]}). {direction} of {abs(change):.2f}%"
            except (TypeError, ValueError):
                return f"❌ Could not compare values for {topic} — invalid or missing numbers in {qtrs[0]} or {qtrs[1]}."

        return f"❌ Could not compare values for {topic}."


    def predict_future(self, topic, question):
        match = re.search(r"q(\d+)", question.lower())
        if match:
            future_quarter = f"q{match.group(1)}"
        else:
            future_quarter = "q5"  

        quarter_num = {f"q{i}": i for i in range(1, 25)}

        if topic not in self.models or topic not in self.best_model_map:
            return f"❌ No model available for {topic}."

        best_model = self.best_model_map[topic]
        model = self.models[topic][best_model]

        q_number = quarter_num.get(future_quarter)
        if not q_number:
            return f"❌ Unsupported quarter: {future_quarter.upper()}."

        if best_model == "average_growth":
            pred = model([[q_number]])[0]
        else:
            pred = model.predict([[q_number]])[0]

        model_name = {
            "linear_regression": "Linear Regression",
            "random_forest": "Random Forest",
            "average_growth": "Average Growth"
        }.get(best_model, best_model)

        return f"{topic.title()}'s {future_quarter.upper()} predicted revenue is {format_currency(pred)} (using {model_name})."



    def answer_query(self, question):
        question = convert_natural_quarter_phrasing(question, current_max_quarter=4)  # You can replace 4 with dynamic max if needed
        complex_response = self.handle_complex_query(question)
        if complex_response is not None:
            return complex_response

        topic, quarter = self.parse_question(question)
        if topic and quarter:
            value = self.get_value(topic, quarter)
            if value:
                return f"{topic} in {quarter.upper()} is {value}."
            else:
                return f"❌ No data available for '{topic}' in '{quarter.upper()}'." 

        if not topic:
            return "❌ Could not identify a valid topic in your question."
        if not quarter:
            return "❌ Could not identify a valid quarter or time period."

        return "❌ Could not understand your query. Please rephrase."


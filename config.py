TOPIC_ALIASES = {
    "ashish": "ashish",
    "faizan": "faizan ali khan",
    "gaurav": "gaurav dharane",
    "kaustubh": "kaustubh a.varde",
    "nitesh": "nitesh jain",
    "prem": "prem prabhanshu",
    "robin": "robin gupta",
    "sanjeev": "sanjeev patni",
    "shariq": "shariq imam",
    "suhail": "suhail",
    "udit": "udit agrawal"
}

QUARTER_MAPPING = {
    "q1": "q1", "first quarter": "q1", "jan-mar": "q1", "first": "q1",
    "q2": "q2", "second quarter": "q2", "apr-jun": "q2", "second": "q2",
    "q3": "q3", "third quarter": "q3", "jul-sep": "q3", "third": "q3",
    "q4": "q4", "fourth quarter": "q4", "oct-dec": "q4", "last quarter": "q4", "last": "q4",
    "current quarter": "q3",
    "next quarter": "q5", "next year": "q5", "next year q1": "q5",
    "future": "q5", "q5": "q5",
    "full year": "sum value", "total": "sum value"
}

MODEL_CONFIG = {
    "model_path": "qa_finetuned",
    "csv_path": "Business Heads.csv",
    "topic_column": "Business Head"
}

import pytest
from qa_pipeline import FinancialQASystem

qa = FinancialQASystem()

# --- BASIC FACTUAL LOOKUP TESTS ---

def test_basic_lookup():
    result = qa.answer_query("What is the value of Ashish in Q1?")
    assert "ashish in q1" in result.lower(), f"Unexpected result: {result}"

def test_alias_lookup():
    result = qa.answer_query("What is the value of udit in Q1?")
    assert "udit agrawal in q1" in result.lower(), f"Unexpected result: {result}"

def test_sum_lookup():
    result = qa.answer_query("What is the total revenue for Kaustubh A. Varde?")
    assert ("kaustubh a.varde in sum value" in result.lower() or 
            "no data" not in result.lower()), f"Unexpected result: {result}"

# --- GROWTH / COMPARISON TESTS ---

def test_percentage_growth():
    result = qa.answer_query("What was the percentage change in Faizan Ali Khan from Q1 to Q2?")
    assert "%" in result, f"Expected percentage in result: {result}"

def test_yes_no_growth():
    result = qa.answer_query("Did Gaurav Dharane increase from Q2 to Q3?")
    assert "yes" in result.lower() or "no" in result.lower(), f"Unexpected result: {result}"

def test_growth_direction():
    result = qa.answer_query("How much did Ashish's revenue change from Q3 to Q4?")
    assert "increased" in result.lower() or "decreased" in result.lower(), f"Unexpected result: {result}"

# --- PREDICTION TESTS ---

def test_predict_q5():
    result = qa.answer_query("Predict next year's q1 revenue for Ashish")
    assert "predicted" in result.lower() and "ashish" in result.lower(), f"Unexpected result: {result}"

def test_predict_q6():
    result = qa.answer_query("What is the estimated revenue for Faizan in the following quarter?")
    assert "predicted" in result.lower() and "faizan" in result.lower(), f"Unexpected result: {result}"

# --- EDGE CASES ---

def test_unknown_topic():
    result = qa.answer_query("What is the value of Ramesh in Q1?")
    assert ("could not identify" in result.lower() or 
            "no data" in result.lower()), f"Unexpected result: {result}"

def test_missing_quarter():
    result = qa.answer_query("How much did Ashish make?")
    assert any(key in result.lower() for key in [
        "could not identify", "no data", "could not find"
    ]), f"Unexpected result: {result}"

def test_ambiguous_question():
    result = qa.answer_query("How is business?")
    assert "could not identify" in result.lower(), f"Unexpected result: {result}"

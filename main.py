from qa_pipeline import FinancialQASystem

def main():
    qa = FinancialQASystem()
    print("CSV Q&A System — type 'exit' to quit.")
    while True:
        q = input("❓ Your question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("🧠", qa.answer_query(q))

if __name__ == "__main__":
    main()

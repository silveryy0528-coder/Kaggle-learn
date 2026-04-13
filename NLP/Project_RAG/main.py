from query import answer_question_with_eval
from src.embedding import load_embedder
from src.llm import load_openai_client


def main():
    artifacts_folder = r'.\data'

    embedder = load_embedder()
    client = load_openai_client('sk-proj-aa')

    question = input('Ask your question: ')

    response = answer_question_with_eval(
        artifacts_folder,
        question,
        embedder,
        client,
        k=5
    )

    print("\nANSWER:\n", response)

if __name__ == "__main__":
    main()
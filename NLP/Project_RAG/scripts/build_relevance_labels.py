#%%
import glob
import sys
from pathlib import Path
sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')

import json
from rapidfuzz import fuzz

from core.ingestion import chunk_multiple_documents

THRESHOLD = 60


def load_qa_dataset(qa_path):
    with open(qa_path, 'r') as f:
        qa_data = json.load(f)
    return qa_data



def find_relevant_chunks(answer, chunks, question=None):
    """
    Weak labeling:
    chunk is relevant if it semantically matches the answer.
    """
    relevant_ids = []

    for chunk in chunks:
        score_answer = fuzz.partial_ratio(answer.lower(), chunk.text.lower())
        score_question = fuzz.partial_ratio(question.lower(), chunk.text.lower())

        score = 1 * score_answer + 0. * score_question

        if score >= THRESHOLD:
            relevant_ids.append(chunk.metadata["chunk_id"])

    if len(relevant_ids) == 0:
        print("NO MATCH")
        print("QUESTION:", question)
        print("ANSWER:", answer)

    return relevant_ids


def main():
    data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data'
    pdf_files = glob.glob(f'{data_folder}/raw/*.pdf')

    chunks = chunk_multiple_documents(pdf_files)

    qa_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\evaluation\datasets\qa_dataset.json"
    output_path = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\eval\relevance_labels.json'

    qa_data = load_qa_dataset(qa_path)

    labeled_data = []

    for item in qa_data:
        answer = item['ground_truth_answer']

        if answer.strip().lower() == 'not found':
            continue

        relevant_chunks = find_relevant_chunks(answer, chunks, question=item["question"])

        labeled_data.append({
            "question": item["question"],
            "answer": answer,
            "relevant_chunk_ids": relevant_chunks,
            "answer_type": item['type']
        })
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(labeled_data, f, indent=2)


if __name__ == "__main__":
    main()
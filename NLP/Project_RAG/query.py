#%%
from src.embedding import load_embedder
from src.retrieval import retrieve_top_k
from src.evaluation import evaluate_answer_llm
from src.llm import build_rag_prompt, generate_answer, load_openai_client
import faiss
import os
import pickle


def load_artifacts(data_folder):
    index_path = os.path.join(data_folder, 'index/faiss.index')
    index = faiss.read_index(index_path)

    chunk_path = os.path.join(data_folder, 'processed/chunks.pkl')
    with open(chunk_path, 'rb') as f:
        chunks = pickle.load(f)

    return index, chunks


def answer_question(folder, question, embedder, client, k=3):
    faiss_index, chunks = load_artifacts(folder)

    top_chunks = retrieve_top_k(question, chunks, embedder, faiss_index, k)
    for c in top_chunks:
        print(f'[Source: {c["doc_id"]}, Page: {c["page"]}]')
        print(c["text"])
        print('-' * 50)

    context = "\n\n".join(c['text'] for c in top_chunks)

    prompt = build_rag_prompt(question, context)

    answer = generate_answer(client, prompt)

    return answer, top_chunks


def answer_question_with_eval(folder, question, embedder, client, k=3):
    answer, top_chunks = answer_question(folder, question, embedder, client, k)

    context = "\n\n".join([c['text'] for c in top_chunks])

    score = evaluate_answer_llm(client, question, answer, context)

    return {
        'question': question,
        'answer': answer,
        'score': score
    }

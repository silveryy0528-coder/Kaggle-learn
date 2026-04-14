#%%
import os
import sys
import faiss
import pickle
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')

import src.embedding as embedding
import src.retrieval as retrieval
import src.llm as llm
import src.ingestion as ingestion
import glob


class RAGService:
    def __init__(self, api_key):
        self.index_path = "data/index/faiss.index"
        self.chunks_path = "data/processed/chunks.pkl"
        self.doc_path = "data/raw/"

        self.embedding_model = embedding.load_embedder()
        self.client = llm.load_openai_client(api_key=api_key)

        self.index, self.chunks = self._load_artefacts()

    def _load_artefacts(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
            return None, None

        index = faiss.read_index(self.index_path)
        with open(self.chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        return index, chunks

    def ask(self, query: str):
        if self.index is None or self.chunks is None:
            return {"error": "Index not built yet. Please upload documents first."}

        top_chunks = retrieval.retrieve_top_k(
            query,
            chunks=self.chunks,
            embedder=self.embedding_model,
            faiss_index=self.index
        )

        context = "\n\n".join(c['text'] for c in top_chunks)
        prompt = llm.build_rag_prompt(query, context)
        answer = llm.generate_answer(self.client, prompt)
        return {
            'query': query,
            'answer': answer,
            'retrieved_chunks': top_chunks
        }

    def rebuild_index(self):
        files = glob.glob(f'{self.doc_path}\\*.pdf')

        chunks = ingestion.chunk_multiple_documents(files)
        texts = [c['text'] for c in chunks]
        embeddings = embedding.embed_text(self.embedding_model, texts)

        index = retrieval.build_faiss_index(embeddings, retrieval.FaissIvfConfig())

        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks, f)

        faiss.write_index(index, self.index_path)

        self.index = index
        self.chunks = chunks

        return {
            'status': 'index built',
            'num_chunks': len(chunks)
        }
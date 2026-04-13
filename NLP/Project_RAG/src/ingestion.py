#%%
import sys
import glob
import os
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
import pymupdf
from src.chunking import chunk_text_with_metadata, ChunkingSentencesConfig


def read_pdf_file(pdf_file):
    doc = pymupdf.open(pdf_file)
    pages = []

    for i, page in enumerate(doc):
        pages.append({
            'text': page.get_text(),
            'page': i + 1
        })

    return pages


def chunk_single_document(pdf_file, chunk_settings=ChunkingSentencesConfig()):
    pages = read_pdf_file(pdf_file)
    doc_id = os.path.basename(pdf_file)
    return chunk_text_with_metadata(pages, doc_id, chunk_settings)


def chunk_multiple_documents(pdf_files, chunk_settings=ChunkingSentencesConfig()):
    all_chunks = []
    global_id = 0

    for pdf_file in pdf_files:
        pages = read_pdf_file(pdf_file)
        doc_id = os.path.basename(pdf_file)
        doc_chunks = chunk_text_with_metadata(pages, doc_id, chunk_settings)

        for chunk in doc_chunks:
            chunk['id'] = global_id
            all_chunks.append(chunk)
            global_id += 1

    return all_chunks


#%%
if __name__ == "__main__":
    pdf_files = glob.glob(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw\*.pdf')
    chunks = chunk_multiple_documents(pdf_files)
    print(chunks[11])
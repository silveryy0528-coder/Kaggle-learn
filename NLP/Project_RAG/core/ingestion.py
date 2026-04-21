#%%
import sys
import glob
import os
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from core.chunking import chunk_text_with_metadata, ChunkingSentencesConfig
import fitz
import re


def clean_text(text):
    text = text.replace("\xa0", " ")      # non-breaking space
    text = text.replace("\t", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r" +", " ", text)
    return text


def is_structure_page(text):
    text = text.strip()
    match = re.match(r"^(\d+)\s+(.+?)(\s+\d+)?$", text)
    if match:
        return True
    return False


def process_page(text_clean):
    return text_clean if not is_structure_page(text_clean) else None


def read_pdf_file(pdf_file):
    doc = fitz.open(pdf_file)
    pages = []

    for i, page in enumerate(doc):
        text_raw = page.get_text("text")
        text_clean = clean_text(text_raw)

        cleaned = process_page(text_clean)
        if cleaned is None:
            continue

        pages.append({
            "text_raw": text_raw,
            "text_clean": text_clean,
            "page": i + 1,
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


data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw'
pdf_files = glob.glob(f'{data_folder}\*.pdf')

documents = read_pdf_file(pdf_files[1])
id = 13
print(documents[id]['text_raw'])
print(documents[id]['text_clean'])

#%%
import sys
import glob
import os
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from core.chunking import (
    chunk_text_with_metadata,
    ChunkingSentenceConfig
)
import fitz
import re
from llama_index.core import Document
from dataclasses import dataclass


@dataclass
class Margin:
    top: int = 50
    bottom: int = 50
    left: int = 50
    right: int = 50


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    text: str


margin = Margin()

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


SPECIAL_SECTIONS = [
    'samenvatting',
    'propositions',
    'stellingen',
    'acknowledgements',
    'about the author',
    'references',
    'copyright'
]

def is_bad_page(text):
    text_low = text.lower()
    return any(k in text_low for k in SPECIAL_SECTIONS)


def process_page(text_clean):
    if is_structure_page(text_clean):
        return None
    elif is_bad_page(text_clean):
        return None
    return text_clean


def read_pdf_file(pdf_file):
    doc = fitz.open(pdf_file)
    doc_id = os.path.basename(pdf_file)

    documents = []
    for i, page in enumerate(doc):
        # 1. Filter out pages with bad structure or content
        text = page.get_text("text")
        cleaned = process_page(text)
        if cleaned is None:
            continue

        # 2. Apply margin filtering
        rect = page.rect
        content_rect = fitz.Rect(
            rect.x0 + margin.left,
            rect.y0 + margin.top,
            rect.x1 - margin.right,
            rect.y1 - margin.bottom
        )
        text = page.get_text("text", clip=content_rect)
        text = clean_text(text)

        document = Document(
            text=text,
            metadata={"page": i + 1, "doc_id": doc_id}
        )
        documents.append(document)

    return documents


def chunk_multiple_documents(pdf_files, chunk_settings=ChunkingSentenceConfig()):
    all_chunks = []
    global_id = 0

    for pdf_file in pdf_files:
        # 1. Read and process the PDF file into Document objects
        pages = read_pdf_file(pdf_file)

        # 2. Chunk the combined text of all pages
        nodes = chunk_text_with_metadata(pages, chunk_settings)

        for node in nodes:
            node = clean_text(node)
            chunk = Chunk(
                chunk_id=global_id,
                doc_id=pages[0].metadata['doc_id'],
                text=node
            )
            all_chunks.append(chunk)
            global_id += 1

    return all_chunks

#%%
if __name__ == "__main__":
    data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw'
    pdf_files = glob.glob(f'{data_folder}\*.pdf')

    chunk_settings = ChunkingSentenceConfig(chunk_size=400, chunk_overlap=50)
    chunks = chunk_multiple_documents(pdf_files, chunk_settings)
    id = 11
    print(chunks[id].text)
    print('--------')
    print(chunks[id+ 1].text)
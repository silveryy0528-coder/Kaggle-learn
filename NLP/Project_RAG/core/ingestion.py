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
from collections import Counter


SPECIAL_SECTIONS = [
    'list of publications',
    'summary',
    'samenvatting',
    'propositions',
    'stellingen',
    'acknowledgements',
    'about the author',
    'references',
    'copyright'
]


@dataclass
class Margin:
    top: int = 50
    bottom: int = 50
    left: int = 50
    right: int = 50

margin = Margin()


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    text: str


def clean_text(text):
    text = text.replace("\xa0", " ")      # non-breaking space
    text = text.replace("\t", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r" +", " ", text)
    return text


def is_content_page(text):
    text = text.strip()
    if '. . . . .' in text:
        return True
    return False


def is_empty_page(text):
    return len(text.strip()) == 0


def is_bad_page(text_clean):
    return (
        is_empty_page(text_clean) or
        is_content_page(text_clean)
    )


def extract_header(page):
    rect = page.rect
    header_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + 40)
    header_text = page.get_text("text", clip=header_rect)
    return header_text.strip().lower()


def extract_section_name(page):
    # 1. Check header for special section names
    header = extract_header(page)
    for section_name in SPECIAL_SECTIONS:
        if section_name in header:
            return section_name

    # 2. If header is empty, check full page text for section names
    empty_header = len(header.strip()) == 0
    if empty_header:
        text = page.get_text("text").lower()
        for section_name in SPECIAL_SECTIONS:
            if section_name in text:
                return section_name
        return 'structural'

    return 'body'


def read_pdf_file(pdf_file):
    doc = fitz.open(pdf_file)
    doc_id = os.path.basename(pdf_file)

    documents = []
    for i, page in enumerate(doc):
        # 1. Filter out pages with bad structure or content
        text = page.get_text("text")
        if is_bad_page(text):
            continue

        section_name = extract_section_name(page)
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
            metadata={
                "page": i + 1,
                "doc_id": doc_id,
                "section": section_name
            }
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

    pages = read_pdf_file(pdf_files[1])
    counter = Counter(page.metadata['section'] for page in pages)
    topk = counter.most_common(115)
    print("Top sections in the PDF:")
    for section, count in topk:
        print(f"  {section}: {count} pages")
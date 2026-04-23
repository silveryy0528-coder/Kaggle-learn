#%%
import sys
import glob
import os
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG')
from core.chunking import (
    chunk_text,
    ChunkingSentenceConfig
)
import fitz
import re
from llama_index.core import Document
from dataclasses import dataclass
from collections import defaultdict


SPECIAL_SECTIONS = [
    'list of publications',
    'summary',
    'propositions',
    'acknowledgements',
    'about the author',
    'samenvatting',
    'stellingen',
    'copyright',
    'references'
]


EXCLUDED_SECTIONS = [
    'samenvatting',
    'stellingen',
    'copyright',
    'references'
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
    text: str
    metadata: dict


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
        if section_name in EXCLUDED_SECTIONS:
            continue

        if 'CV' in doc_id and section_name == 'structural':
            section_name = 'body'

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


def group_pages_by_section(pages):
    sections = defaultdict(list)
    for page in pages:
        section_name = page.metadata['section']
        sections[section_name].append(page)
    return sections


def chunk_single_document(
        pdf_file,
        chunk_settings=ChunkingSentenceConfig(),
        chunk_id_offset=0):
    # 1. Read and process the PDF file into Document objects
    pages = read_pdf_file(pdf_file)
    doc_id = pages[0].metadata['doc_id']

    # 2. Group pages by section in {'section_name': [list of pages]}
    sections = group_pages_by_section(pages)

    all_chunks = []
    chunk_id = chunk_id_offset
    for section_name, section_pages in sections.items():
        print(f"Processing {doc_id} - Section: {section_name} with {len(section_pages)} pages")
        nodes = chunk_text(section_pages, chunk_settings)

        for node in nodes:
            node = clean_text(node)
            chunk = Chunk(
                text=node,
                metadata={
                    "section": section_name,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id}
            )
            chunk_id += 1
            all_chunks.append(chunk)

    return all_chunks


def chunk_multiple_documents(pdf_files, chunk_settings=ChunkingSentenceConfig()):
    all_chunks = []
    id_offset = 0

    for pdf_file in pdf_files:
        chunks = chunk_single_document(pdf_file, chunk_settings, id_offset)
        id_offset += len(chunks)
        all_chunks.extend(chunks)

    return all_chunks


#%%
if __name__ == "__main__":
    data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw'
    pdf_files = glob.glob(f'{data_folder}\*.pdf')

    chunk_settings = ChunkingSentenceConfig(chunk_size=500, chunk_overlap=50)
    chunks = chunk_multiple_documents(pdf_files, chunk_settings)
    idx = 1
    print(chunks[idx].metadata)
    print(chunks[idx].text[-500:])
    print('-------------')
    print(chunks[idx + 1].metadata)
    print(chunks[idx + 1].text[:500])

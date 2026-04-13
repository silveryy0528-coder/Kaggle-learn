#%%
from dataclasses import dataclass
import re
import os
import sys
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\src')
import ingestion


@dataclass
class ChunkingConfig():
    chunk_size: int = 400


@dataclass
class ChunkingSentencesConfig(ChunkingConfig):
    sentence_limit: int = 50


def _split_text_to_sentences(text, limit=50):
    sentences = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            print(line)
            continue
        if len(line) > limit:
            parts = [p.strip() for p in re.split(r'\.\s+', line) if p.strip()]
            sentences.extend(parts)
        else:
            sentences.append(line)
    return sentences


def chunk_text_sentences(text, chunk_size=400, sentence_limit=50):
    sentences = _split_text_to_sentences(text, sentence_limit)

    chunks = []
    current_chunk = ""
    for sent in sentences:
        sent = sent.strip()

        if len(current_chunk) + len(sent) < chunk_size:
            if current_chunk:
                current_chunk += (" " + sent)
            else:
                current_chunk = sent
        else:
            chunks.append({"text": current_chunk})
            current_chunk = sent

    if current_chunk:
        chunks.append({"text": current_chunk})

    return chunks


def chunk_text(text, settings):
    if not isinstance(settings, ChunkingConfig):
        raise TypeError('Wrong chunking settings provided.')

    chunk_size = settings.chunk_size
    if isinstance(settings, ChunkingSentencesConfig):
        return chunk_text_sentences(text, chunk_size, settings.sentence_limit)


def chunk_text_with_metadata(text, source, settings):
    raw_chunks = chunk_text(text, settings)

    chunks = []
    for i, raw_chunk in enumerate(raw_chunks):
        chunk = {
            'id': i,
            'text': raw_chunk['text'],
            'doc_id': source,
            'page': 1
        }
        chunks.append(chunk)

    return chunks


#%%
if __name__ == "__main__":
    pdf_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\raw\CV_YanGuo.pdf"
    source = os.path.basename(pdf_file)
    text = ingestion.read_pdf_file(pdf_file)
    chunks = chunk_text_with_metadata(text, source, ChunkingSentencesConfig())
    print(chunks[-2])
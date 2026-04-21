#%%
from dataclasses import dataclass
import re
import sys
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\src')


@dataclass
class ChunkingConfig():
    chunk_size: int = 500


@dataclass
class ChunkingSentencesConfig(ChunkingConfig):
    sentence_limit: int = 100


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


def chunk_text_with_metadata(pages, doc_id, settings):
    chunks_with_metadata = []
    chunk_id = 0

    for page_data in pages:
        text = page_data['text']
        page_number = page_data['page']
        raw_chunks = chunk_text(text, settings)

        for raw_chunk in raw_chunks:
            chunk_with_metadata = {
                'id': chunk_id,
                'text': raw_chunk['text'],
                'doc_id': doc_id,
                'page': page_number
            }
            chunk_id += 1
            chunks_with_metadata.append(chunk_with_metadata)

    return chunks_with_metadata

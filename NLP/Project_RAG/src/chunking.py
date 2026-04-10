#%%
from dataclasses import dataclass
import re


@dataclass
class ChunkingConfig():
    chunk_size: int = 400


@dataclass
class ChunkingNaiveConfig(ChunkingConfig):
    pass


@dataclass
class ChunkingOverlapConfig(ChunkingConfig):
    overlap: int = 100


@dataclass
class ChunkingSentencesConfig(ChunkingConfig):
    sentence_limit: int = 50


def chunk_text_naive(text, chunk_size=400):
    '''
    Basic chunking without overlap. Chunks can be cut mid-sentence and mid-word.
    '''
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


def chunk_text_overlap(text, chunk_size=400, overlap=100):
    '''
    Chunking with overlap
    '''
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks


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
        if len(current_chunk) + len(sent) < chunk_size:
            if current_chunk:
                current_chunk += (" " + sent)
            else:
                current_chunk = sent
        else:
            chunks.append(current_chunk)
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text(text, settings):
    if not isinstance(settings, ChunkingConfig):
        raise TypeError('Wrong chunking settings provided.')

    chunk_size = settings.chunk_size
    if isinstance(settings, ChunkingNaiveConfig):
        return chunk_text_naive(text, chunk_size)
    elif isinstance(settings, ChunkingOverlapConfig):
        return chunk_text_overlap(text, chunk_size, settings.overlap)
    elif isinstance(settings, ChunkingSentencesConfig):
        return chunk_text_sentences(text, chunk_size, settings.sentence_limit)

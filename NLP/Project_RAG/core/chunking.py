#%%
from dataclasses import dataclass
import re
import sys
sys.path.insert(0, r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\src')
from llama_index.core.node_parser import SentenceSplitter


#%%
@dataclass
class ChunkingConfig():
    chunk_size: int = 500


@dataclass
class ChunkingSentenceConfig(ChunkingConfig):
    chunk_overlap: int = 50


def sentence_splitter(documents, chunk_size, chunk_overlap):
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(documents)

    return nodes


def chunk_text_with_metadata(documents, settings):
    nodes = sentence_splitter(documents, settings.chunk_size, settings.chunk_overlap)
    return nodes


if __name__ == "__main__":
    pass
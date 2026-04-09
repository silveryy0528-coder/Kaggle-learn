#%%
#%%
import openai
openai.api_key = 'YOUR_KEY'
import pymupdf
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_pdf(pdf_file):
    doc = pymupdf.open(pdf_file)
    for page in doc:
        text = page.get_text()

    return text


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


def chunk_text_sentences(text, chunk_size=400):
    sentences = _split_text_to_sentences(text)

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


def chunk_text(text, method, args):
    if method == 'naive':
        return chunk_text_naive(text, **args)
    elif method == 'overlap':
        return chunk_text_overlap(text, **args)
    elif method == 'sentence':
        return chunk_text_sentences(text, **args)
    else:
        raise TypeError(f'Unsupported method "{method}".')


def embed_text(embedder, chunks):
    return embedder.encode(
        chunks,
        device='cuda',
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def build_faiss_flat(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def faiss_search(index, query_vecs, k=3):
    scores, indices = index.search(query_vecs, k)
    return scores, indices


def grounding_score(answer, context):
    answer_tokens = re.findall(r"\w+", answer.lower())
    context_tokens = set(re.findall(r"\w+", context.lower()))

    if not answer_tokens:
        return 0

    overlap = [t for t in answer_tokens if t in context_tokens]

    score = len(overlap) / len(answer_tokens)
    return score


def _llm_call(model_name, prompt, role="user", temperature=1):
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": role, "content": prompt}
        ],
        temperature=temperature
    )
    return response


def evaluate_answer_quality(question, answer, context, model_name='gpt-4.1-mini'):
    # Word overlap
    score = grounding_score(answer, context)

    # LLM-based evaluation
    prompt = f"""
    You are evaluating a QA system.

    Question:
    {question}

    Context:
    {context}

    Answer:
    {answer}

    Evaluation rules:
    1. If the context contains the answer, the answer must match it.
    2. If the context does NOT contain the answer, the correct response is "Not found".
    3. If the answer says "Not found" and the context indeed lacks the information, this is CORRECT.
    4. If the answer contains information not in the context, it is INCORRECT.

    Is the answer correct?

    Return in this format:
    Result: YES or NO
    Reason: <short explanation>
    """
    response = _llm_call(model_name, prompt)

    return score, response.choices[0].message.content


def evaluate_retrieval_quality(question, retrieved_chunk, model_name='gpt-4.1-mini'):
    prompt = f"""
    Given the question and retrieved chunk

    Question:
    {question}

    Retrieved chunk:
    {retrieved_chunk}

    Is this chunk relevant for answering the question?
    ONLY answer YES or NO
    """
    response = _llm_call(model_name, prompt)

    return response.choices[0].message.content


def retrieve_answer_from_llm(question, context, model_name='gpt-4.1-mini'):
    prompt = f"""
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "Not found".

    Context:
    {context}

    Question:
    {question}
    """
    response = _llm_call(model_name, prompt)

    return response.choices[0].message.content


#%%
pdf_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\NLP\Project_RAG\data\CV_YanGuo.pdf"

model_name="all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name, device='cuda')

#%% Build retrieval pipeline
text = read_pdf(pdf_file)
args = {'chunk_size': 400}
chunks = chunk_text(text, method='sentence', args=args)
embeddings = embed_text(embedder, chunks=chunks)
faiss_index = build_faiss_flat(embeddings)

#%% Embed query and perform context retrieval
queries = [
    'Has the candidate worked with machine learning?',
    'Has the candidate ever worked with FEM?'
]
query_vecs = embed_text(embedder, queries)

k = 3
all_scores, all_indices = faiss_search(faiss_index, query_vecs, k=k)
for query_idx, (retrieval_scores, retrieval_indices) in enumerate(zip(all_scores, all_indices)):
    question = queries[query_idx]
    print(f'Query {query_idx + 1} - {question}')

    for top_idx in range(k):
        print(70 * '-')
        retrieved_chunk = chunks[retrieval_indices[top_idx]]
        print(
            f'Top {top_idx + 1} - Score {retrieval_scores[top_idx]:.2f}\n'
            f'"{retrieved_chunk}"')

#%% Send the query and retrieved context to LLM and unpack the answer
for query_idx, retrieval_indices in enumerate(all_indices):
    question = queries[query_idx]
    retrieved_chunks = [chunks[i] for i in retrieval_indices]
    context = "\n\n".join(retrieved_chunks)

    answer = retrieve_answer_from_llm(question, context)

    overlap_score, llm_eval = evaluate_answer_quality(question, answer, context)

    print(
        70 * '-',
        f'\nQ: {question}\n',
        f'A: {answer}\n',
        f'Quality check - {llm_eval}'
    )



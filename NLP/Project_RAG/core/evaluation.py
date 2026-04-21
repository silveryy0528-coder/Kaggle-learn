import re
from src.llm import generate_answer


def grounding_score(answer, context):
    answer_tokens = re.findall(r"\w+", answer.lower())
    context_tokens = set(re.findall(r"\w+", context.lower()))

    if not answer_tokens:
        return 0

    overlap = [t for t in answer_tokens if t in context_tokens]
    return len(overlap) / len(answer_tokens)


def evaluate_answer_llm(
        client, question, answer, context, model_name='gpt-4.1-mini', temperature=0):
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
    return generate_answer(client, prompt, model_name, temperature)


def evaluate_retrieval_llm(
        client, question, chunk, model_name='gpt-4.1-mini', temperature=0):
    prompt = f"""
Given the question and retrieved chunk

Question:
{question}

Retrieved chunk:
{chunk}

Is this chunk relevant for answering the question?
ONLY answer YES or NO
"""
    return generate_answer(client, prompt, model_name, temperature)
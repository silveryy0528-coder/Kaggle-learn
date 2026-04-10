from openai import OpenAI


def load_openai_client(api_key=None):
    return OpenAI(api_key=api_key)


def generate_answer(client, prompt, model_name='gpt-4.1-mini', temperature=1):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": 'user', "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def build_rag_prompt(question, context):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found".

Context:
{context}

Question:
{question}
"""
    return prompt
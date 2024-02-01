from openai import OpenAI

def prompt_gpt(client: OpenAI, prompt: str, model: str = "gpt-4-1106-preview") -> str:
    completion = client.chat.completions.create(messages=[{"role": "user",
                                                           "content": prompt}],
                                                model=model)
    return completion.choices[0].message.content.replace("\n\n", " ")
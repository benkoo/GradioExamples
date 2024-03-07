import ollama


stream = ollama.chat(
    model='llama2',
    messages = [{'role':'user','content': '您能說中文嗎?'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
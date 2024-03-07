import faiss
import requests
import numpy as np

d = 4096 # The dimension count for each embedding is 4096


titles = [
    "Hello and good bye",
    "Some useful statements",
    "Silicon valley Starups finds new way to preduct Protein folding",
    "中文可以嗎？",
    "Team XYZ Wins World Cup in Rugby",
    "Biology: One of the most underrated college majors",
    "Novel Machine Leanrning algorithm changes neuroscience forever",
    "California is rising in the west"
    ]

index = faiss.IndexFlatL2(d)

X = np.zeros((len(titles),d), dtype='float32')

for i, title in enumerate(titles):
    res = requests.post(
                url ='http://localhost:11434/api/embeddings',
                json={
                    'model' :'llama2',
                    'prompt': title
                })
    embedding = res.json()['embedding']
    X[i] = np.array(embedding)

print(len(res.json()['embedding']))
#print(res.json()['embedding'])

#new_prompt = 'California companies on the rise'
new_prompt = 'How is your Chinese Language?'
#new_prompt = "Greetings"
#new_prompt = "Fighter gest UFC title"
new_prompt = "What are the Protein related Startups in Silicon Valley?"

res = requests.post(
                    url ='http://localhost:11434/api/embeddings',
                    json={
                        'model' :'llama2',
                        'prompt': new_prompt
                    })

index.add(X)

embedding = np.array(object=[res.json()['embedding']], dtype='float32')

D, I = index.search(embedding, 5)

print(np.array(titles)[I.flatten()])


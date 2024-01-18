
# RAG
RAG(Retriever-Augmented Generation)는 데이터베이스에서 관련 정보를 검색(retrieve)하여 생성(generate)하는 작업을 돕는 기술입니다. 

간단한 RAG 모델을 구현하기 위해, 먼저 벡터 데이터베이스와 언어 모델을 준비해야 합니다. 이 예제에서는 Faiss라는 META의 효율적인 벡터 검색 라이브러리와 Hugging Face의 Transformers를 사용하겠습니다.

1. 필요한 라이브러리를 설치합니다.
```python
pip install faiss-cpu transformers sentence-transformers
```

2. 벡터 데이터베이스를 생성합니다. 여기서는 Wikipedia 문서를 예시로 사용하겠습니다.
```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ['위키백과 문서1 내용', '위키백과 문서2 내용', '위키백과 문서3 내용', ...]
embeddings = model.encode(sentences)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype('float32'))
```

3. 간단한 sLLM 언어 모델을 이용해 생성합니다. 여기서는 Hugging Face의 GPT2를 사용하겠습니다.
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
```

4. 입력 문장에 대해 벡터 데이터베이스에서 가장 가까운 벡터를 찾고, 해당 벡터를 사용해 언어 모델을 생성합니다.
```python
input_sentence = '입력 문장'
input_embedding = model.encode([input_sentence])

D, I = index.search(np.array(input_embedding).astype('float32'), k=1)
retrieved_sentence = sentences[I[0][0]]

input_ids = tokenizer.encode(retrieved_sentence, return_tensors='pt')
generated_output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)

generated_sentence = tokenizer.decode(generated_output[0], skip_special_tokens=True)
```
RAG 모델은 입력 문장과 검색된 문장을 함께 사용해 생성할 수 있는 구조입니다.

--------------------------


# PDF -> Graph DB -> Vector DB -> sLLM

1. PDF 문서를 텍스트로 변환하기
```python
from PyPDF2 import PdfFileReader

def extract_text_from_pdf(file_path):
    pdf = PdfFileReader(file_path)
    text = ""
    for page in range(pdf.getNumPages()):
        text += pdf.getPage(page).extractText()
    return text

file_path = 'path_to_your_pdf_file.pdf'
text = extract_text_from_pdf(file_path)
```

2. 텍스트에서 개체명과 관계 추출하기 (여기서는 Spacy 라이브러리를 사용하였습니다)
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)
```

3. 추출된 개체명과 관계를 지식 그래프 데이터베이스에 저장하기 (여기서는 Neo4j를 사용하였습니다)
```python
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

def add_entity_to_db(tx, name, label):
    tx.run("CREATE (a:Label {name: Name})", Label=label, Name=name)

with driver.session() as session:
    for entity in doc.ents:
        session.write_transaction(add_entity_to_db, entity.text, entity.label_)
```

4. 질문이 들어오면 지식 그래프에서 관련 정보를 검색하고, 이를 RAG 모델의 입력으로 사용하기
```python
def find_related_entity(tx, name):
    result = tx.run("MATCH (a) WHERE a.name = Name RETURN a", Name=name)
    return [record["a"]["name"] for record in result]

question = 'your_question'
with driver.session() as session:
    related_entities = session.read_transaction(find_related_entity, question)

# related_entities를 RAG 모델의 입력으로 사용
```

--------------------------


지식 그래프는 노드와 엣지로 구성된 구조화된 데이터셋입니다. 
노드는 엔티티를 나타내고, 엣지는 엔티티 간의 관계를 나타냅니다. 
이를 통하여 벡터 DB에 저장된 정보에 추가적인 맥락을 제공하여 검색의 정확성을 높일 수 있습니다.

1. 지식 그래프를 생성합니다. 이를 위해 텍스트에서 개체명과 관계를 추출하고, 이를 그래프 형태로 저장합니다. 이 과정은 자연어 처리와 그래프 데이터베이스를 함께 사용하게 됩니다.
```python
# 지식 그래프 생성 예시
import spacy
from neo4j import GraphDatabase

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

def add_entity_to_db(tx, name, label):
    tx.run("CREATE (a:Label {name: Name})", Label=label, Name=name)

with driver.session() as session:
    for entity in doc.ents:
        session.write_transaction(add_entity_to_db, entity.text, entity.label_)
```

2. 사용자의 입력과 관련된 정보를 지식 그래프에서 검색하고, 이를 벡터 DB에서 가장 가까운 벡터를 찾는데 사용합니다.
```python
# 지식 그래프에서 관련 정보 검색 예시
def find_related_entity(tx, name):
    result = tx.run("MATCH (a) WHERE a.name = Name RETURN a", Name=name)
    return [record["a"]["name"] for record in result]

question = 'your_question'
with driver.session() as session:
    related_entities = session.read_transaction(find_related_entity, question)

# related_entities를 벡터 DB 검색에 사용
input_embedding = model.encode(related_entities)
D, I = index.search(np.array(input_embedding).astype('float32'), k=1)
```

위의 코드는 지식 그래프와 벡터 DB를 조합하는 방법입니다. 
지식 그래프에서 검색한 정보를 벡터 DB에서 가장 가까운 벡터를 찾는데 사용했고, 검색 결과를 RAG 모델에 입력으로 제공하기 위해 검색된 문장을 언어 모델에 입력으로 사용하여 새로운 문장을 생성합니다.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT2모델 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 벡터 DB에서 검색된 문장을 사용해 생성
retrieved_sentence = sentences[I[0][0]]
input_ids = tokenizer.encode(retrieved_sentence, return_tensors='pt')
generated_output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)

# 생성된 문장 출력
generated_sentence = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print(generated_sentence)
```
지식 그래프에서 검색된 정보를 이용해 새로운 문장을 생성하는 과정입니다. 
이 과정은 RAG 모델의 핵심 원리를 반영하고 있으나 실제 비즈 단에서는 고도화 및 전문성을 요합니다. POC를 마칩니다.

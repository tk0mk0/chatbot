open_api_key = ''
pinecone_api_key = ''
pinecone_environment = ''
### Load preprocessed data

# from datasets import load_dataset
# import sys
#
# sys.setrecursionlimit(2500)
# data = load_dataset("csv",data_files=r'C:\Users\toko\Desktop\Reviews_new.csv')
# print('data loaded')
### tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding('cl100k_base')


#create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

#Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=30,
    length_function=tiktoken_len,
    separators=["\n\n", "\n"]
)
print('tokenizer')

### Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    document_model_name=model_name,
    query_model_name=model_name,
    openai_api_key=open_api_key
)
print('embedding function')

###VectorDB
import pinecone
index_name = 'testchatbot'
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)
print('Vectordb Initiated')

## Used to create index
# pinecone.create_index(
#     name=index_name,
#     metric='dotproduct',
#     dimension=1536  # 1536 dim of text-embedding-ada-002
# )
# print('index created')

index = pinecone.GRPCIndex(index_name)
print('connected to pinecone index')

### UPSERT DATA FOR INDEX
from tqdm.auto import tqdm
from uuid import uuid4

# batch_limit = 100
# texts = []
# metadatas = []
#
# for i, record in enumerate(tqdm(data['train'])):
#     # first get metadata fields for this record
#     metadata = {
#         'product': str(record['ProductId']),
#         'score': record['Score'],
#         'summary': record['Summary']
#     }
#     # now we create chunks from the record text
#     record_texts = text_splitter.split_text(record['combined'])
#     # create individual metadata dicts for each chunk
#     record_metadatas = [{
#         "chunk": j, "text": text, **metadata
#     } for j, text in enumerate(record_texts)]
#     # append these to current batches
#     texts.extend(record_texts)
#     metadatas.extend(record_metadatas)
#     # if we have reached the batch_limit we can add texts
#     if len(texts) >= batch_limit:
#         ids = [str(uuid4()) for _ in range(len(texts))]
#         embeds = embed.embed_documents(texts)
#         index.upsert(vectors=zip(ids, embeds, metadatas))
#         texts = []
#         metadatas = []
#print('data inserted in pinecone')


### connect pinecone to langchain
from langchain.vectorstores import Pinecone
text_field = "text"
# switch back to normal index for langchain
index = pinecone.Index(index_name)
vectorstore = Pinecone(
    index, embed.embed_query, text_field)
retr = vectorstore.as_retriever(search_kwargs={"k": 4})

print('connected langchain to pinecone')

from langchain import OpenAI
from langchain.chains import RetrievalQA


### LLM and Agents
llm = OpenAI(
    openai_api_key='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',
    model_name='text-davinci-003',
    temperature=0.0,
    max_tokens=512,
    frequency_penalty=1
)
context_answer = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retr,
    verbose = True
)

print('context_answer initiated')



### New db answer
from langchain.prompts.prompt import PromptTemplate
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the following tables:
reviews

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
)
from langchain import SQLDatabase, SQLDatabaseChain
db = SQLDatabase.from_uri('sqlite:///chatbot.sqlite')
db_answer=SQLDatabaseChain(llm=llm, database=db, prompt = PROMPT, verbose=True)
print('db_answer initiated')

### Agent and toolls
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
tools = [
    Tool(
        name = "context",
        func=context_answer.run,
        description="useful for when you need to answer context questions. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
        verbose=True

    ),
    Tool(
        name = "db",
        func=db_answer.run,
        description="useful for when you need to answer database questions. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
        verbose=True
    ),
]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print('agent initiated')


query = 'text question. what are the opinions of people about dog food?'


### get context documents
test_rez = vectorstore.similarity_search(query, k=4)


result = agent.run(query)
print('done')
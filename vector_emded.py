import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

DEPLOYMENET_NAME = "textEmbedding3Large"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.9
)

loader = PyPDFLoader(".\data\Claim Closure Letter.pdf")
docs = loader.load()
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap= 10,
    length_function= len,
    add_start_index = True
)  

splits = text_splitter.split_documents(docs)
# print(len(splits))

emdeddings = AzureOpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),
                                   azure_endpoint="https://digital-ai-assistance.openai.azure.com/",
                                   api_version="2024-02-01",
                                   azure_deployment=DEPLOYMENET_NAME)

persist_directory = ".\data\db\chroma"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=emdeddings,
    persist_directory=persist_directory
)

#print(vectorstore._collection.count())

query = "what is the status?"
docs_resp = vectorstore.similarity_search(query=query,k=4)
#print(len(docs_resp))
print(docs_resp[1].page_content)
print(docs_resp[0].page_content)
print(docs_resp[2].page_content)
print(docs_resp[3].page_content)
#vectorstore.persist()


# text1="Math is great subject to study"
# text2="Dogs are friendly when they are happy and feed well"
# text3="Physics is not one of my favorites subjects"

# embed1 = emdeddings.embed_query(text=text1)
# embed2= emdeddings.embed_query(text=text2)
# embed3 = emdeddings.embed_query(text=text3)

# # print(f"Embed1 == {embed1}")

# similarity = np.dot(embed1,embed2)
# print(f"similarity %:{similarity*100}")
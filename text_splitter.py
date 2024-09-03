import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.9
)

with open(".\data\I-have-a-dream.txt") as paper:
    speech = paper.read()
    
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap= 20,
    length_function= len
)    
    
text = text_splitter.create_documents([speech])  
print(text[0])  
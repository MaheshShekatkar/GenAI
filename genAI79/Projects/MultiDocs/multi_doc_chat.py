import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from load_docs import load_docs
from langchain.chains import ConversationalRetrievalChain

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.9
)

#pdf_loader = PyPDFLoader(".\docs\pythonftopeningnotes.pdf")
documents = load_docs()
chat_history =[]

text_splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

docs = text_splitter.split_documents(documents=documents)
vector = Chroma.from_documents(
    documents=docs,
    embedding= AzureOpenAIEmbeddings(),
    persist_directory='.\data\db\chroma'
)

# qa_chain = ConversationalRetrievalChain.
import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import azure_openai
from langchain_community.llms import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import chroma
from langchain_openai import AzureOpenAI  
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

DEPLOYMENET_NAME = "gpt-35-turbo"

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

os.environ["OPENAI_API_VERSION"] = client.openai_api_version
os.environ["AZURE_OPENAI_ENDPOINT"] = client.azure_endpoint

load_dotenv(find_dotenv())
prompt = "How old is the universe"
messages = [HumanMessage(content=prompt)]
print(client.invoke("Write a python function to add 2 number"))
print("===============")
#print(chatClient.invoke("What is the weather in WA DC"))
print(chatClient.predict_messages(messages=messages).content)

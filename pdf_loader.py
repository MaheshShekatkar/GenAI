import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.9
)

loader = PyPDFLoader(".\data\Microservies.pdf")
pages = loader.load()
page = pages[3]
#print(len(pages))
print(page.page_content[0:700])
print(page.metadata)
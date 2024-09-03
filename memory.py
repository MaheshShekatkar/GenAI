import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.6
)

#print(chatClient.invoke("My name is kapil. What is yours?"))
#print(chatClient.invoke("Great! What is my name?"))

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chatClient,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hello there, I am Kapil")
conversation.predict(input="Why sky is blue")
conversation.predict(input="Whats my name?")

print(memory.load_memory_variables({'AI'}))
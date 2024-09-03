import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

DEPLOYMENET_NAME = "gpt-35-turbo"

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

# chatClient = AzureChatOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version="2024-02-01",
#     azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
#     deployment_name = DEPLOYMENET_NAME,
#     temperature= 0.9
# )

template = """
As a children's book writer , please come up with a simple a short (90 words)
lullaby based on location
{location}
and main character {name}

STORY:
"""

prompt = PromptTemplate(input_variables=["location","name"],
                        template=template)
chain_story = LLMChain(llm=client,prompt=prompt)
story = chain_story({"location": "Zanzibar","name":"Maya"})

print(story['text'])
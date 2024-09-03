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
from langchain.prompts import ChatPromptTemplate

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

os.environ["OPENAI_API_VERSION"] = chatClient.openai_api_version
os.environ["AZURE_OPENAI_ENDPOINT"] = chatClient.azure_endpoint

load_dotenv(find_dotenv())

def get_completion(prompt):
    message=[{"role":"user","content":prompt}]
    response= chatClient.predict_messages(messages=message)
    
    return response.content

customer_review = """This product is garbage. It broke after one use. 
Don't bother wasting your money. Horrible experience."""

tone ="""polite tone"""
language = "Urdu"
prompt = f"""
Rewrite the folllowig {customer_review} in the {tone}, 
and then please translate the new review message in the {language}.
"""
rewrite = get_completion(prompt=prompt)
print(rewrite)

# ===== using langchain & prompt template - Still chat API 
template_string = """
Translate the following text {customer_review}
into italiano in polite tone
and the company name {company_name}
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
translation_message = prompt_template.format_messages(
    customer_review = customer_review,
    company_name = "Amazon"
)

response = chatClient.invoke(translation_message)
print(response.content)
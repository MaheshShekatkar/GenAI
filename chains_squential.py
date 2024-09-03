import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

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
chain_story = LLMChain(llm=client,prompt=prompt,
                       output_key="story",
                       verbose=True)
story = chain_story({"location": "Zanzibar","name":"Maya"})

# print(story['text'])

template_update= """
Translate the {story} into {language}.Make sure 
the language is simple and fun 

TRANSLATION:
"""

prompt_Template = PromptTemplate(input_variables=["story","language"],
                                 template=template_update)
chain_translate = LLMChain(
    llm=client,
    prompt=prompt_Template,
    output_key ="translated"
)

overall_chain = SequentialChain(
    chains= [chain_story,chain_translate],
    input_variables=["location","language","name"],
    output_variables= ["story","translated"],
    verbose= True
)

response = overall_chain({"location": "Zanzibar",
                          "name":"Maya",
                          "language":"French"})

print(f"English verson ====> {response['story']} \n \n ")
print(f"Translated verson ====> {response['translated']} \n \n ")
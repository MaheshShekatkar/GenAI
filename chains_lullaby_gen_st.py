import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st

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



def generate_lullaby(location,name,language):
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
    # story = chain_story({"location": "Zanzibar","name":"Maya"})
    # print(story['text'])

    template_update= """
    Translate the {story} into {language}.Make sure 
    the language is simple and fun 

    TRANSLATION:
    """

    prompt_template = PromptTemplate(input_variables=["story","language"],
                                    template=template_update)
    chain_translate = LLMChain(
        llm=client,
        prompt=prompt_template,
        output_key ="translated"
    )

    overall_chain = SequentialChain(
        chains= [chain_story,chain_translate],
        input_variables=["location","language","name"],
        output_variables= ["story","translated"],
        verbose= True
    )

    response = overall_chain({"location": location,
                            "name": name,
                            "language":language})

    # print(f"English verson ====> {response['story']} \n \n ")
    # print(f"Translated verson ====> {response['translated']} \n \n ")
    
    return response


def main():
    st.set_page_config(page_title="Generate children lullaby"
                       ,layout="centered")
    st.title("Let AI Write and Translate the lullaby for you 📖")
    st.header("Get Started...")
    
    location_input = st.text_input(label="Where is the story set?")
    main_character_input = st.text_input(label="What's the main character's name?")
    language_input = st.text_input(label="Translate story into...")
    
    submit_button = st.button("Submit")
    if location_input and main_character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby..."):
                response = generate_lullaby(location=location_input,
                                            name=main_character_input,
                                            language=language_input)
                with st.expander("English version"):
                    st.write(response['story'])
                with st.expander(f"{language_input} version"):
                    st.write(response['translated'])    

if __name__ == '__main__':
    main()
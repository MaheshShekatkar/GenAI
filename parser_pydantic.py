import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel,Field,validator
from typing import List

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

class OfficeInfo(BaseModel):
    reach_time: str = Field(description="when person reach the office")
    hybrid_mode_days: str = Field(description="In which the days person go to office ")
    timing: List = Field(description="different timing mentioned in the email")
    
    
pydantic_parser = PydanticOutputParser(pydantic_object=OfficeInfo)
format_instructions = pydantic_parser.get_format_instructions()

email_response ="""
I am working in hybrid mode. Going office on Tuesday, Wednesday & Thursday. I reach office on 10.30 AM.
Attend daily meeting on 11.00 AM. I go for lunch on 12.30 PM. Leave from office at 5.30 PM.
"""

email_template = """
from following email , extract the following information

email:{email}
{format_instructions}
"""

updated_prompt = ChatPromptTemplate.from_template(template=email_template)
message = updated_prompt.format_messages(email=email_response,
                                         format_instructions=format_instructions)
format_response = chatClient.invoke(message)
print(type(format_response.content))

vacation = pydantic_parser.parse(format_response.content)
print(type(vacation))
print(vacation.timing)
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

DEPLOYMENET_NAME = "gpt-35-turbo"

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

email_response ="""
I am working in hybrid mode. Going office on Tuesday, Wednesday & Thursday. I reach office on 10.30 AM.
Attend daily meeting on 11.00 AM. I go for lunch on 12.30 PM. Leave from office at 5.30 PM.
"""
email_template = """
from following email , extract the following information
reach_time: when person reach the office
hybrid_mode_days: In which the days person go to office 
timing: different timing mentioned in the email

format the ouput as JSON with the following keys
reach_time
hybrid_mode_days
timing

email:{email}
{format_instructions}
"""

#prompt_template = ChatPromptTemplate.from_template(email_template)
#print(prompt_template)
#message = prompt_template.format_messages(email=email_response)
#response = chatClient.invoke(message)
#print(response.content)

reach_time_schema = ResponseSchema(name="reach_time",
                                   description="Office reach time")
hybrid_mode_days_schema = ResponseSchema(name="hybrid_mode_days",
                                         description="Days on which person go to office in hybrid mode")
timing_schema = ResponseSchema(name="timing",
                               description="timing present in the text")

response_schema = [
    reach_time_schema,
    hybrid_mode_days_schema,
    timing_schema
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)
format_instructions = output_parser.get_format_instructions()
#print(format_instructions)
updated_prompt = ChatPromptTemplate.from_template(template=email_template)
parser_message = updated_prompt.format_messages(email=email_response,
                                          format_instructions=format_instructions)
parser_response = chatClient.invoke(parser_message)
print(type(parser_response.content))

output_dict = output_parser.parse(parser_response.content)
print(output_dict)
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.router.multi_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain

DEPLOYMENET_NAME = "gpt-35-turbo"

# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version="2024-02-01",
#     azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
#     deployment_name = DEPLOYMENET_NAME,
#     temperature= 0.7
# )

chatClient = AzureChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.9
)

biology_template = """
Photosynthesis is a fundamental biological process by which green plants, algae, 
and certain bacteria convert light energy into chemical energy. Using sunlight, 
these organisms transform carbon dioxide and water into glucose and oxygen. 
This process not only fuels the organism's own energy needs but also produces oxygen, 
which is essential for the respiration of most living creatures on Earth.

Here is the qustion:
{input}
"""

math_template = """
The Pythagorean Theorem is a fundamental principle in geometry that applies to right-angled triangles. 
It states that in a right triangle, the square of the length of the hypotenuse 
(the side opposite the right angle) is equal to the sum of the squares of the lengths of the other 
two sides.

Here is the question:
{input}
"""

astronomy_template = """
Black holes are regions of space where the gravitational pull is so strong that nothing, 
not even light, can escape from them. They are formed when massive stars collapse under 
their own gravity at the end of their life cycle.
"""

travel_agent_template = """
A top travel agent is a professional who specializes in planning, booking, and managing travel 
experiences for individuals and groups. They offer personalized services, such as creating 
customized itineraries, securing the best deals on flights, accommodations, and transportation, 
and providing expert advice on destinations, activities, and travel logistics. 
Here is the question:
{input}
"""
prompt_infos=[
    {
        "name":"Biology",
        "description":"Good for answering the Biology related questions",
        "prompt_template": biology_template
    },
    {
        "name":"Math",
        "description":"Good for answering the Math related questions",
        "prompt_template": math_template
    },
    {
        "name":"Astronomy",
        "description":"Good for answering the Astronomy related questions",
        "prompt_template": astronomy_template
    },
     {
        "name":"Travel_agent",
        "description":"Good for answering the Travel Agent related questions",
        "prompt_template": travel_agent_template
    }
]

destination_chains = {}
for info in prompt_infos:
    name = info["name"]
    prompt_template = info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=chatClient,prompt=prompt)
    destination_chains[name]= chain
    
# print(f"DESTINATION...{destination_chains}")   
deafult_prompt = ChatPromptTemplate.from_template("{input}")
deafult_chain = LLMChain(llm=chatClient,prompt=deafult_prompt)

destinations = [f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
print(destinations_str)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)

router_chain = LLMRouterChain.from_llm(
    chatClient,
    router_prompt
)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=deafult_chain,
    verbose=True
)

response = chain.invoke("what is cash balance?")
#response = chain.run("When did the Ramayan happen")
print(response)

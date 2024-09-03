import os
import requests
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

DEPLOYMENET_NAME = "gpt-35-turbo"

# Define a function to call the external API
def get_weather(city: str) -> str:
    # Replace with your actual API URL and API key
    api_key = "6e55c4d2f9144848818112113241808"
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"

    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return "Sorry, I couldn't find the weather for that location."

    current_weather = data['current']
    temp_c = current_weather['temp_c']
    condition = current_weather['condition']['text']

    return {
        "city": city,
        "temp_c": temp_c,
        "condition": condition
    }

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

# Define a function to handle the user input and generate appropriate responses
# def custom_prompt_handler(user_input: str, chat_history: dict, memory: ConversationBufferMemory) -> str:
#     if "weather" in user_input.lower():
#         city = user_input.split("in")[-1].strip()  # Simple heuristic to extract city name
#         weather_info = get_weather(city)
        
#         if "error" in weather_info:
#             return weather_info["error"]
        
#         # Store the temperature in memory
#         memory.add_message("temp_c", weather_info["temp_c"])
#         memory.add_message("city", weather_info["city"])
        
#         return f"The current temperature in {city} is {weather_info['temp_c']}°C and the weather is {weather_info['condition']}."
    
#     if "fahrenheit" in user_input.lower():
#         # Retrieve the temperature in Celsius from memory
#         print("custom handler call............")
#         temp_c = memory.get_memory()["temp_c"]
#         if temp_c is not None:
#             # Convert Celsius to Fahrenheit
#             temp_f = (temp_c * 9/5) + 32
#             return f"The temperature in {memory.get_memory()['city']} is {temp_f}°F."
#         else:
#             return "I don't have the temperature information to convert. Please ask for the weather first."
    
#     return "How can I assist you?"

# class CustomLLMChain(LLMChain):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _call(self, inputs: dict) -> str:
#         user_input = inputs["user_input"]
#         print(f"from cutom logic:::{user_input}")
#         chat_history = inputs.get("chat_history", "")
#         # Call the custom handler to fetch weather or respond to other queries
#         response = custom_prompt_handler(user_input, chat_history, self.memory)
        
#         # Update the memory with the new conversation turn
#         self.memory.add_message(user_input, response)
        
#         return response


# Create a tool for the agent to use
weather_tool = Tool(
    name="Weather Tool",
    func=get_weather,
    description="Get the current weather for a given city."
)

# Define the prompt template
prompt_template = """
You are an assistant that helps users with weather information.
You can fetch the weather for a given city and store it in memory.
you can provide the temperature for the provided city.
If the user asks for the temperature in Fahrenheit after receiving the weather in Celsius, convert it.

User: {user_input}
Assistant:"""

# Define the prompt template for the LLM to use
prompt = ChatPromptTemplate.from_template(prompt_template)

translation_message = prompt.format_messages(
    customer_review = prompt_template,
    user_input = {input}
)


# Initialize the language model (you can use other LLMs as well)
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint ="https://digital-ai-assistance.openai.azure.com/",
    deployment_name = DEPLOYMENET_NAME,
    temperature= 0.7
)

# Create an LLM chain with the prompt and model
llm_chain = LLMChain(
    llm=client,
    prompt=prompt,
    memory=memory,
)

# Initialize the agent with the LLM chain and the tool
agent = initialize_agent(
    tools=[weather_tool],
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    llm_chain=llm_chain,
    llm=client,
    memory=memory,
    max_iterations = 2,
    verbose=True
)

# Test the agent with a user query
response_1 = agent.invoke({"input":"What's the weather like in NYC in Fahrenheit?"})
#response_2 =agent("what is price in Indian rupee of Iphone 12?")
print(f"response:::{response_1['output']}")
#print(f"response:::{response_2}")

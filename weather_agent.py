# Import necessary components from LangChain and Ollama

from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import PromptTemplate
import os

# --- Configuration ---
# Ensure Ollama is running and 'tinyllama' model is pulled.
# You can pull the model using: ollama pull tinyllama
OLLAMA_MODEL = "tinyllama"
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama server address

# --- Initialize Ollama LLM ---
# The Ollama class allows LangChain to interact with the Ollama server.
# We specify the model and the base URL of the Ollama server.
try:
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"Successfully initialized Ollama with model: {OLLAMA_MODEL}")
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    print("Please ensure Ollama is running and the 'tinyllama' model is downloaded.")
    print("You can download the model by running: ollama pull tinyllama")
    exit() # Exit if Ollama cannot be initialized


# --- Define a simple tool for the agent ---
# Tools are functions that the agent can call to perform specific actions.
# This example tool simply returns a predefined string.
def get_current_weather(location: str) -> str:
    """
    Returns the current weather for a given location.
    This is a dummy function for demonstration purposes.
    """
    if "london" in location.lower():
        return "It's cloudy with a chance of rain in London, 15°C."
    elif "new york" in location.lower():
        return "It's sunny and warm in New York, 25°C."
    else:
        return f"Weather information for {location} is not available."

# Create a LangChain Tool object from the Python function.
# The 'name' is how the agent will refer to the tool.
# The 'description' helps the LLM understand when to use the tool.
tools = [
    Tool(
        name="get_weather",
        func=get_current_weather,
        description="Useful for getting the current weather for a specific location. Input should be a city name.",
    )
]

# --- Define the Agent Prompt ---
# The prompt guides the LLM on how to act as an agent,
# how to use tools, and how to format its thoughts and actions.
# This template uses the ReAct (Reasoning and Acting) pattern.
prompt_template = PromptTemplate.from_template("""
You are a helpful AI assistant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# --- Create the Agent ---
# `create_react_agent` constructs an agent that uses the ReAct pattern.
# It requires the LLM, the list of tools, and the prompt template.
agent = create_react_agent(llm, tools, prompt_template)

# --- Create the Agent Executor ---
# The AgentExecutor is responsible for running the agent.
# It takes the agent and the tools, and can be configured with verbosity
# to see the agent's thought process.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Run the Agent ---
# print("\n--- Running Agent with a question about weather ---")
# try:
#     result = agent_executor.invoke({"input": "What is the weather like in London?"})
#     print("\nAgent's Final Answer:")
#     print(result["output"])
# except Exception as e:
#     print(f"\nAn error occurred during agent execution: {e}")

# print("\n--- Running Agent with a question about an unknown location ---")
# try:
#     result = agent_executor.invoke({"input": "What is the weather like in Tokyo?"})
#     print("\nAgent's Final Answer:")
#     print(result["output"])
# except Exception as e:
#     print(f"\nAn error occurred during agent execution: {e}")

print("\n--- Running Agent with a general question (should not use tool) ---")
try:
    result = agent_executor.invoke({"input": "Tell me a fun fact about cats."})
    print("\nAgent's Final Answer:")
    print(result["output"])
except Exception as e:
    print(f"\nAn error occurred during agent execution: {e}")

# conversational_agent_example.py
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from langchain.agents import load_tools, Tool
from langchain.tools import DuckDuckGoSearchResults

from langchain.agents import AgentExecutor, create_react_agent

if __name__ == "__main__":
    # --- Configuration ---
    OLLAMA_MODEL = "tinyllama"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # --- Initialize Ollama LLM ---
    try:
        llm = Ollama(
            model=OLLAMA_MODEL, 
            base_url=OLLAMA_BASE_URL,
            temperature=0.01, # Try a lower temperature
            num_predict=512 # Equivalent to max_new_tokens in Ollama
        )
        print(f"Initialized Ollama LLM with model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Please ensure Ollama is running and 'tinyllama' model is downloaded.")
        exit()


    react_template = """Answer the following questions as best you can.
            You have access to the following tool: {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: I need to use a tool to answer this question.
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat if needed)
            Thought: I now know the final answer.
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}"""


    prompt = PromptTemplate(
        template=react_template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )

    
    # You can create the tool to pass to an agent
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="duckduck",
        description="A web search engine. Use this to as a search engine for general queries.",
        func=search.run,
    )

    # Prepare tools
    # tools = load_tools(["llm-math"], llm=llm)
    tools=[search_tool]

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5 # Set a reasonable limit for the number of steps

    )

    # --- Test Cases ---
    print("\n--- Running Agent Example 1 ---")
    try:
        agent_executor.invoke({"input": "What is the weather in Munich?"})
    except Exception as e:
        print(f"Agent execution failed for 'What is the weather in Munich?': {e}")

    print("\n--- Running Agent Example 2 ---")
    try:
        agent_executor.invoke({"input": "What is the capital of Germany?"})
    except Exception as e:
        print(f"Agent execution failed for 'What is the capital of Germany?': {e}")
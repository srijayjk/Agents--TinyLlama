from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory


# 🧠 Use TinyLLaMA via Ollama
llm = Ollama(model="tinyllama")

# 🛠 Python tool to execute code
python_tool = PythonREPLTool()

# 🔌 Wrap the tool as a LangChain Tool
tools = [
    Tool.from_function(
        func=python_tool._run,
        name="PythonREPL",
        description="Executes Python code and returns the result."
    )
]

# 🧠 Add memory so it remembers the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🤖 Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 🧪 Example interaction
response = agent.run("What is 5*12? Then plot a simple matplotlib chart.")
print("Agent says:\n", response)

# conversational_agent_example.py
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    '''
    Explanation:
    The ConversationBufferMemory stores the ongoing dialogue. 
    Each time conversation.invoke() is called, the prompt_template is populated with the accumulated history 
    (the previous turns), allowing the LLM to maintain context and refer back to earlier parts of the conversation.
    '''
    # --- Configuration ---
    OLLAMA_MODEL = "tinyllama"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # --- Initialize Ollama LLM ---
    try:
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"Initialized Ollama LLM with model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Please ensure Ollama is running and 'tinyllama' model is downloaded.")
        exit()

    # --- Initialize Conversation Memory ---
    memory = ConversationBufferMemory()

    # --- Define the Conversational Prompt ---
    # The {history} variable is crucial for the agent to remember past turns.
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template="""The following is a friendly conversation between a human and an AI.
    The AI is remembers everything and responds with specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI:"""
    )

    # --- Create the Conversational Chain ---
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt_template,
        verbose=True # Set to True to see the prompt with history
    )

    # --- Run the Conversational Agent ---
    print("\n--- Running Conversational Agent ---")

    print("\nHuman: Hi there!")
    response = conversation.invoke({"input": "Hi there!"})
    print(f"AI: {response['response']}")

    print("\nHuman: My name is Alice.")
    response = conversation.invoke({"input": "My name is Alice."})
    print(f"AI: {response['response']}")

    print("\nHuman: What did I just tell you my name was?")
    response = conversation.invoke({"input": "What did I just tell you my name was?"})
    print(f"AI: {response['response']}")

    print("\nHuman: What is your favorite color?")
    response = conversation.invoke({"input": "What is your favorite color?"})
    print(f"AI: {response['response']}")
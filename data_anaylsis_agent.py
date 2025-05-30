import pandas as pd
import os
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced plotting
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Keep MessagesPlaceholder for reference if needed, though not directly used by initialize_agent's default prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
CSV_FILE_PATH = "sample_data.csv"
OLLAMA_MODEL = "tinyllama" # Ensure this model is pulled in Ollama (e.g., ollama pull tinyllama)

# Define df globally so it can be accessed by the tool
df = None

def get_CSV_data():
    """Loads or creates a sample CSV file and returns its DataFrame."""
    global df # Declare intent to modify the global df
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Creating sample CSV file: {CSV_FILE_PATH}")
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Age': [24, 30, 22, 35, 28],
            'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Houston'],
            'Salary': [70000, 90000, 60000, 110000, 85000]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(CSV_FILE_PATH, index=False)
        print("Sample CSV created successfully.")
    else:
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            print(f"Successfully loaded CSV into DataFrame. Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading CSV file '{CSV_FILE_PATH}': {e}")
            exit()
    return df

def get_tinyllama_1b():
    """Initializes and returns the ChatOllama LLM."""
    try:
        llm = ChatOllama(model=OLLAMA_MODEL)
        print(f"Successfully connected to Ollama with model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"Error connecting to Ollama or loading model '{OLLAMA_MODEL}': {e}")
        print("Please ensure Ollama is running and you have 'tinyllama' pulled (e.g., `ollama pull tinyllama`).")
        exit() # Exit if LLM cannot be loaded
    return llm

# --- 4. Define a Custom Tool for Pandas DataFrame Operations with Plotting ---
@tool
def python_repl_pandas(code: str) -> str:
    """Executes pandas, matplotlib, and seaborn code on a global DataFrame named 'df' and returns the output.
    The 'df' variable refers to the DataFrame loaded from the CSV file.
    Always print the results of your code execution to stdout.
    If you generate a plot, you MUST save it to a file named 'plot.png' using `plt.savefig('plot.png')`
    and then clear the current figure using `plt.clf()` to prepare for subsequent plots.
    For example: print(df.head()) or print(df['Age'].mean()) or plt.hist(df['Age']); plt.savefig('age_hist.png'); plt.clf()
    """
    try:
        global df # Access the global DataFrame
        # Redirect stdout to capture print statements
        import io
        import sys
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        # Pass df, pd, plt, and sns into the execution context for the LLM to use
        exec(code, {'df': df, 'pd': pd, 'plt': plt, 'sns': sns})

        sys.stdout = old_stdout # Restore stdout
        output = redirected_output.getvalue()
        return output if output else "Execution successful, no direct output (check if you used print())."
    except Exception as e:
        return f"Error executing code: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    # List of tools available to the agent
    tools = [python_repl_pandas]
    
    # Initialize global df and llm
    df = get_CSV_data()
    llm = get_tinyllama_1b()

    # Get the schema of the DataFrame to include in the agent's prefix
    df_schema_markdown = df.head().to_markdown(index=False)
    
    # Define a prefix to guide the LLM more effectively and explicitly
    # This prefix will be passed to initialize_agent via agent_kwargs
    agent_prefix = f"""You are an AI assistant specialized in analyzing tabular data using Pandas, and visualizing it with Matplotlib and Seaborn.
        You have access to a Pandas DataFrame named 'df', which contains the data from 'sample_data.csv'.
        You must use the 'python_repl_pandas' tool to execute Python code for any data analysis or visualization tasks.
        Always print the results of your code execution to stdout.
        If you generate a plot, you MUST save it to a file named 'plot.png' using `plt.savefig('plot.png')` and then clear the current figure using `plt.clf()`.

        Here is the schema of the DataFrame:
        {df_schema_markdown}

        You should always follow this exact format for your responses:

        Question: the input question you want to answer
        Thought: you should always think about what to do, and why you are choosing a particular action.
        Action: the action to take, should be one of {python_repl_pandas}
        Action Input: the input to the action (Python code to execute, ensure it's valid and executable)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times until you have the final answer)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!
        """

    # --- Create the Agent Executor using initialize_agent ---
    try:
        agent_executor = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, # Set verbose=True to see the agent's detailed reasoning steps
            handle_parsing_errors=True, # Good for debugging if the LLM output doesn't match expected format
            agent_kwargs={"prefix": agent_prefix}, # Correctly pass the custom prefix here
            # Crucial for smaller models: Add stop sequences to prevent malformed output
            stop=["\nObservation:", "\nThought:", "\nFinal Answer:"]
        )
        print("Agent Executor created successfully using initialize_agent with custom prefix and stop sequences.")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        exit()

    # --- Interact with the Agent ---
    print("\n--- CSV Data Analysis Assistant (Built with initialize_agent) ---")
    print("Ask me questions about the data in 'sample_data.csv'.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour query: ")
        if user_query.lower() == 'exit':
            print("Exiting Data Analysis Assistant. Goodbye!")
            break

        try:
            response = agent_executor.invoke({"input": user_query})
            print(f"\nAgent's Final Answer: {response['output']}")
            # Inform the user if a plot was likely generated
            if "plot.png" in response['output'] or any("plot.png" in str(step) for step in response.get('intermediate_steps', [])):
                 print("\nNote: If a plot was generated, check your script's directory for 'plot.png'.")
        except Exception as e:
            print(f"An error occurred while processing your query: {e}")
            print("Please try rephrasing your question or check the console for more details.")


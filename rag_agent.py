# rag_agent_example.py
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# --- Configuration ---
OLLAMA_MODEL = "tinyllama"
OLLAMA_BASE_URL = "http://localhost:11434"


def get_tinyllama_1b():
    """Initializes and returns the ChatOllama LLM."""
    try:
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"Initialized Ollama LLM and Embeddings with model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"Error connecting to Ollama or loading model '{OLLAMA_MODEL}': {e}")
        print("Please ensure Ollama is running and you have 'tinyllama' pulled (e.g., `ollama pull tinyllama`).")
        exit() # Exit if LLM cannot be loaded

    return llm, embeddings

def get_document():
    # --- Create a simple document store ---
    documents = [
        Document(page_content="The capital of France is Paris. Paris is known for the Eiffel Tower."),
        Document(page_content="Mount Everest is the highest mountain in the world, located in the Himalayas."),
        Document(page_content="Python is a popular programming language for AI and data science."),
        Document(page_content="The Earth revolves around the Sun.")
    ]
    return documents

if __name__ == "__main__":
    llm, embeddings = get_tinyllama_1b()
    documents = get_document()


    # Split documents into smaller chunks (optional for small docs, but good practice)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(documents)

    # Create a vector store from the documents
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("Created in-memory vector store.")

    # --- Create the RAG Chain ---
    # We use RetrievalQA chain to combine retrieval and LLM generation
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" means simply stuffing all retrieved documents into the prompt
        retriever=vectorstore.as_retriever(),
        return_source_documents=True # Optionally return the documents that were used
    )

    # --- Run the RAG Agent ---
    print("\n--- Running RAG Agent ---")

    question1 = "What is the highest mountain?"
    print(f"\nQuestion: {question1}")
    response = qa_chain.invoke({"query": question1})
    print(f"Answer: {response['result']}")
    print(f"Source Documents: {[doc.page_content for doc in response['source_documents']]}")

    question2 = "Where is the Eiffel Tower?"
    print(f"\nQuestion: {question2}")
    response = qa_chain.invoke({"query": question2})
    print(f"Answer: {response['result']}")
    print(f"Source Documents: {[doc.page_content for doc in response['source_documents']]}")

    question3 = "What is the best color?" # A question not in our documents
    print(f"\nQuestion: {question3}")
    response = qa_chain.invoke({"query": question3})
    print(f"Answer: {response['result']}")
    print(f"Source Documents: {[doc.page_content for doc in response['source_documents']]}")
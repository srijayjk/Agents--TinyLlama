# Agent-DataAnalysis-TinyLlama


# LangChain Agents with Ollama (TinyLlama) Examples

This repository contains simple Python examples demonstrating various types of AI agents built using LangChain, leveraging the local power of Ollama with the `tinyllama` model. Each example showcases a different agent paradigm, illustrating how LangChain components can be combined to create intelligent, task-oriented systems.

## Prerequisites

Before running any of the examples, ensure you have:

1. **Ollama Installed:** Download and install Ollama from [ollama.com](https://ollama.com/).
2. **TinyLlama Model Pulled:** Open your terminal and run `ollama pull tinyllama`.
3. **Python Dependencies:** Install the necessary LangChain libraries:
   ```bash
   pip install langchain langchain-community langchain-core pydantic
   ```

---

## Projects Overview

### 1. Retrieval-Augmented Generation (RAG) Agent

* **Description:** This agent specializes in answering questions by retrieving relevant information from a predefined knowledge base (documents) and then generating a coherent response based on the retrieved context using the LLM.
* **Key Components:** `Ollama` (LLM & Embeddings), `Chroma` (Vector Store), `RecursiveCharacterTextSplitter`, `RetrievalQA` chain.
* **Use Cases:** Building Q&A systems over custom datasets, chatbots that can cite sources, enterprise search.
* **Code:** `rag_agent_example.py`

### 2. Tool-Using Agent

* **Description:** An agent capable of performing actions by intelligently selecting and using external "tools" (Python functions) based on the user's query. This example includes a simple arithmetic calculator tool.
* **Key Components:** `Ollama` (LLM), `Tool`, `create_react_agent`, `AgentExecutor`, `PromptTemplate` (ReAct pattern).
* **Use Cases:** Automating tasks, integrating with APIs, data manipulation, interacting with external systems.
* **Code:** `tool_using_agent_example.py`

### 3. Conversational Agent

* **Description:** This agent maintains a memory of past interactions, allowing it to engage in more natural, context-aware conversations by recalling previous turns.
* **Key Components:** `Ollama` (LLM), `ConversationChain`, `ConversationBufferMemory`, `PromptTemplate` (with history).
* **Use Cases:** Chatbots, virtual assistants, customer support agents that remember user preferences or past queries.
* **Code:** `conversational_agent_example.py`

### 4. Planning Agent

* **Description:** Designed to break down complex user requests into smaller, manageable sub-tasks. It systematically uses tools to execute these steps, working towards a final answer.
* **Key Components:** `Ollama` (LLM), `Tool`, `create_react_agent`, `AgentExecutor`, `PromptTemplate` (emphasizing step-by-step thinking).
* **Use Cases:** Multi-step problem-solving, automating workflows requiring sequential actions, complex data gathering.
* **Code:** `planning_agent_example.py`

### 5. Autonomous Agent (Simplified Concept)

* **Description:** A highly simplified conceptual example illustrating the iterative nature of an autonomous agent. It attempts to achieve a broad goal by performing actions and "reflecting" on progress over multiple turns.
* **Key Components:** `Ollama` (LLM), `Tool` (simulated tasks/status), `create_react_agent`, `AgentExecutor`, `PromptTemplate` (goal-oriented, self-reflection).
* **Use Cases:** (In full scale) Self-correcting systems, continuous task execution, complex research, creative content generation.
* **Code:** `autonomous_agent_concept.py`
  * *Note: This example is illustrative. Fully autonomous agents involve more complex state management and evaluation loops.*

### 6. Multi-Agent Systems (Conceptual)

* **Description:** This section conceptually explains systems where multiple specialized agents collaborate, each handling a specific part of a larger problem by communicating and passing tasks/information to one another.
* **Key Components:** (Conceptual) Multiple `AgentExecutor` instances, distinct `Tool` sets per agent, and an orchestration layer for communication.
* **Use Cases:** Complex simulations, distributed problem-solving, collaborative AI workflows (e.g., a researcher AI passing findings to a writer AI).
* **Code:** *Conceptual sketch provided in the previous response, not a standalone runnable script due to inherent complexity.*

---

## How to Run

To run any example, navigate to the directory containing the script and execute:

```bash
python your_agent_example.py
```

# -Multi-Agent-System
This is a Python-based multi-agent system designed to handle complex tasks through a hierarchical structure of AI agents. It leverages Large Language Models (LLMs) via the Groq API for core reasoning and code generation, ChromaDB for persistent memory, and Langchain for agent and tool management.

The system can understand natural language requests, delegate tasks, generate Python code, execute it, handle errors through a debugging sub-agent, and remember past interactions to inform future ones.

## Features

*   **Multi-Agent Architecture:**
    *   **Heart (LLM):** Central LLM for task analysis, code generation, and information retrieval.
    *   **Main Agent:** Strategic planner, task decomposer, and delegator.
    *   **Parent AI Agent:** Executes code, manages retries, and can delegate to child agents for debugging.
    *   **Child Agent (Debugger):** Analyzes execution errors and provides reports.
*   **Code Generation & Execution:** Generates Python code to fulfill requests and executes it using `PythonREPLTool` or a custom `run_exec` tool (with safety caveats).
*   **Persistent Memory:** Utilizes ChromaDB with HuggingFace embeddings to store and recall information, including successful code snippets and past interactions, for both individual agents and a shared memory pool.
*   **Error Handling & Correction:** A dedicated child agent analyzes errors, and the Heart LLM attempts to fix the code based on the analysis.
*   **Tool Integration:**
    *   Web search (DuckDuckGo)
    *   Python code execution
    *   File saving
    *   Custom delegation tool
*   **Dynamic Workflow:** The system can recall past interactions to enrich current task processing.

## Architecture Overview

1.  **User Input:** A task is provided to the `workflow` function.
2.  **Memory Recall:** The `AgentLogger` (for the Main Agent) attempts to recall relevant past plans or interactions from ChromaDB.
3.  **Task Delegation (`delegate_task`):**
    *   Checks memory for existing solutions.
    *   If no suitable memory, it consults the `Main_Agent` for a plan and then the `Heart` LLM (via `chain`) to generate code.
    *   The `Parent_AI_Agent` is tasked with executing the generated code.
    *   Successful code and its task description are stored in shared memory.
4.  **Error Handling:**
    *   If execution fails, the `Child1_Agent` analyzes the error.
    *   The `Heart` LLM receives the error report and attempts to generate corrected code.
    *   The `Parent_AI_Agent` attempts to execute the corrected code.
5.  **Output:** The final result (either successful output or an error message after recovery attempts) is returned.

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   Git

### Installation Steps

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <https://github.com/akshayabalan>
    cd <https://github.com/akshayabalan/-Multi-Agent-System>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    langchain-core
    langchain-groq
    langchain
    langchain-experimental
    langchain-community
    langchain-chroma
    chromadb
    sentence-transformers
    duckduckgo-search
    groq
    # Optional, but good practice for API key management
    # python-dotenv
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Key Configuration:**
    The script currently has the Groq API key hardcoded:
    `groq_api_key="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"`

    **IMPORTANT:** Replace this with your actual Groq API key. It is strongly recommended to manage API keys using environment variables instead of hardcoding them.
    For example, you could use a `.env` file:
    ```
    GROQ_API_KEY="your_actual_groq_api_key_here"
    ```
    And load it in your script using a library like `python-dotenv`:
    ```python
    # At the beginning of your script
    from dotenv import load_dotenv
    import os
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model_name="Llama3-70b-8192",
        groq_api_key=GROQ_API_KEY, # Use the loaded key
        temperature=0.1,
        max_tokens=6000,
    )
    ```

## How to Run

The script `prototype2.py` is currently set up to run a predefined task:
`given_input = "Write some interesting facts and create a file and store it in the file."`

To run the script:
```bash
python prototype2.py

The script will print the final result of the workflow to the console.
A memory_db directory will be created in the same location as the script to store the ChromaDB persistent data.
Key Components & Files
prototype2.py: The main script containing all agent definitions, tools, memory management, and the workflow logic.
memory_db/ (directory): Created automatically by ChromaDB to persist agent memories. Can be safely deleted if you want to reset memory, but it will be recreated on the next run.
Agents:
llm (Heart): ChatGroq instance for core LLM interactions.
Main_Agent: Orchestrates tasks and delegates.
Parent_AI_Agent: Executes code and manages sub-tasks.
Child1_Agent: Analyzes errors in code execution.
Memory:
MemoryManager: Class to manage ChromaDB instances.
AgentLogger: Class for individual agent memory and interaction with shared memory.
Tools:
run_exec: Executes arbitrary Python code. Use with extreme caution due to security risks.
DuckDuckGoSearchRun: Performs web searches.
save_to_file: Saves content to a specified file.
PythonREPLTool: Standard Langchain tool for Python code execution.
Delegate (custom tool): Wraps the delegate_task function for the Main Agent.
Important Notes & Caveats
Security Risk (run_exec tool): The run_exec tool allows arbitrary Python code execution. This is a significant security risk if the system is exposed to untrusted input. Use with extreme caution and consider sandboxing or replacing with safer alternatives for production environments.
API Key Security: Ensure your Groq API key is kept confidential and not committed to public repositories. Use environment variables or a secure secret management system.
Model Dependency: The system is configured to use Llama3-70b-8192 via Groq. Performance and behavior may vary with other models.
Memory Persistence: Agent memories are stored in the memory_db directory. This allows the system to retain knowledge across sessions.
Failsafe Mechanism: The delegate_task function includes a basic failsafe to prevent potential infinite delegation loops.
Future Improvements / TODOs
Implement more robust API key management (e.g., require .env file).
Enhance the security of code execution (e.g., sandboxing environment for run_exec).
Develop a more sophisticated task planning and decomposition mechanism in the Main_Agent.
Expand the toolset available to agents.
Add more comprehensive error handling and recovery strategies.
Implement unit and integration tests.
Allow for dynamic configuration of agents and models (e.g., via a config file).
Explore UI or API interfaces for interacting with the system.


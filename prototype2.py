from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentType , initialize_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb 
from langchain.tools import tool
import shutil
import inspect
def get_current_code():
    with open(__file__, "r") as f:
        return f.read()
llm = ChatGroq(
    model_name="Llama3-70b-8192",
    groq_api_key="Chat_Groq_Key",
    temperature=0.1,  
    max_tokens=6000,  
)
output_parsor = StrOutputParser()


@tool
def run_exec(code: str) -> str:
    """
    Executes arbitrary Python code (use with extreme caution).
    
    This tool runs Python code provided in the `code` argument and returns the result. 
    If an error occurs during execution, it returns the error message.
    """
    try:
        local_vars = {}  # Executes arbitrary Python code (use with extreme caution)
        exec(code, {}, local_vars)
        return str(local_vars)
    except Exception as e:
        return f"ERROR: {e}"

risky_tools = [
    run_exec,
    
]
prompt = ChatPromptTemplate.from_messages([
    ("system", """**Role**: You are the **Heart** - the central LLM powering a multi-agent system. Your job is to:  
1. **Analyze tasks** from the Main Agent,  
2. **Generate code/info**, and  
3. **Route responses correctly** between agents.  

**Rules**:  
1. **Task Classification**:  
   - If the task is **informational** (e.g., "Explain recursion"), respond with:  
     ```info\n[concise answer]\n```  
   - If the task is **code-related** (e.g., "Write Python code to sort a list"), respond with:  
     ```code\n[language]\n[code]\n```  

2. **Error Handling**:  
   - If the Parent Agent reports an error (e.g., ```error\n[language]\n[error_message]\n```), you **must**:  
     a) Fix the code,  
     b) Return it in the same ```code\n...``` format.  

3. **Safety & Clarity**:  
   - Reject dangerous requests (e.g., "rm -rf") with:  
     ```error\nREQUEST_REJECTED: [reason]\n```  
   - If the task is unclear, ask:  
     ```info\nClarify: [specific question]\n```  

**Examples**:  
1. **Information Request**:  
   - Input: "Explain how HTTPS works"  
   - Output: ```info\nHTTPS encrypts data...\n```  

2. **Code Request**:  
   - Input: "Python code to reverse a string"  
   - Output: ```code\npython\ndef reverse_string(s): return s[::-1]\n```  

3. **Error Recovery**:  
   - Input: ```error\npython\nIndexError: list index out of range\n```  
   - Output: ```code\npython\nfixed_code_here\n```"""),
    ("user", "{task}")
])

chain = prompt | llm



class MemoryManager :
    def __init__(self,path = "memory_db") :
        self.embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        self.path = path
        os.makedirs(path,exist_ok =True) # this Safely creates nested directories without checking if they exist first.
        self.client = chromadb.PersistentClient(path=path) 
    def get_agent_memory(self,agent_name : str) :
        # each agent must have their own memory
        return Chroma(
            collection_name = agent_name,
            client=self.client,
            embedding_function= self.embedding,
            persist_directory = self.path)
memory = MemoryManager() 

class AgentLogger :
    def __init__(self,agent_name) :
        self.log = []
        self.memory = memory.get_agent_memory(agent_name)
        self.shared_memory = memory.get_agent_memory("shared")   # common pool
    def remember (self,text : str , is_shared) :
        target = self.shared_memory if is_shared else self.memory
        target.add_texts([text])
    def recall(self, query,k=3, search_shared : bool =True ) :
        private_results = self.memory.similarity_search_with_score(query, k = k)
        if search_shared :
           shared_results = self.shared_memory.similarity_search_with_score(query , k=k)
           return private_results + shared_results
        return private_results
   

# Global logger  Chroma
logger = AgentLogger("main_agent")

def is_valid_python_code(text: str) -> bool:
    return any(keyword in text for keyword in ['def ', 'import ', 'print(', 'class ', 'if __name__'])

def delegate_task(task: str, delegation_attempted=False) -> str:
    print(f"[Delegation Started] Task Received: {task}")

    recall_results = logger.recall(f"Code for: {task}", k=1)
    if recall_results:
        doc, score = recall_results[0]
        try:
            score = float(score)
        except ValueError:
            score = 0.0

        print(f"Memory Recall Score: {score}")
        
        # Correct threshold logic — and bail if score is high
        if score <= 1.5 and score > 0.85:
            content = doc.page_content.strip()
            if is_valid_python_code(content):
                print("[Delegation] Using recalled memory.")
                return content

    if delegation_attempted:
        print("[Failsafe] Delegation already attempted. Preventing infinite loop.")
        return "FAILSAFE: Delegation unsuccessful. Responding without further delegation."

    print("[Delegation] No suitable memory found. Asking LLM.")
    
    # ask LLM via Main Agent to understand the task 
    plan = Main_Agent.invoke({"input": f"Ask the LLM instruction on how to do the {task}"})

    # get the actual code from the LLM
    written_code = chain.invoke({"task": task})

    # ask Parent Agent to execute it                 
    execution_result = Parent_AI_Agent.invoke({
        "input": f"""Execute this Python code:\n```python\n{written_code}\n```"""
    })

    if "error" not in execution_result.lower():
        logger.remember(f"Code for: {task}\n{written_code}", is_shared=True)
        print("[Delegation] Execution successful and memory stored.")
    else:
        print("[Delegation] Execution failed; error detected.")

    return execution_result

main_tools = [ DuckDuckGoSearchRun(name="web_search"),
    Tool(
        name="Delegate",
        func = delegate_task,
        description="Delegates task to LLM for code generation, and to agents for execution and error handling"
    )
]
Main_Agent = initialize_agent(
    llm = llm,
    tools = main_tools ,
    verbose = True,
    handle_parsing_errors=True, 
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs = { "system_message" : """
You are the **Main AI Agent** – the central intelligence coordinating an AI agent system. Your responsibilities:

1. **Strategic Planning**:
   - Break complex goals into executable tasks
   - Decide which sub-agent (Parent/Children) should handle each task
   - Example: "To scrape a website, first generate code -> validate -> execute"

2. **Agent Coordination**:
   - Delegate code generation to the LLM (Heart)
   - Assign execution to Parent Agent
   - Route debugging to Children Agents

3. **Behavior Rules**:
   - For SIMPLE tasks (printing, file operations, etc.) execute directly using available tools
   - Only delegate COMPLEX tasks requiring code generation
   - NEVER delegate:
     * Basic print/output operations
     * Simple file operations
     * Web search result formatting
    -* Zero Memory Dependence**: Ignore past interactions unless explicitly provided in the current task.  
    -* No Loops**: If a task fails twice, escalate to human.

4. **Output Format**:
   Task -> Target Agent -> Requirements  
   Example:  
   "TASK: Create star.txt -> PARENT_AGENT -> Requires: File content 'I never knew...'"""
,"memory" : AgentLogger("Main_Agent"), "user" : "{input}"}

)

@tool
def save_to_file(input_str: str) -> str:
    """
    Save content to a file.
    Expected input format: "filename::CONTENT"
    """
    try:
        if "::" not in input_str:
            return "ERROR: Input format must be 'filename::content'"
        filename, content = input_str.split("::", 1)
        full_path = os.path.abspath(filename.strip())
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File saved at: {full_path}"
    except Exception as e:
        return f"ERROR: {e}"

parent_tools = [PythonREPLTool() , run_exec , save_to_file]
Parent_AI_Agent = initialize_agent (
        tools = parent_tools,
        llm = llm,
        handle_parsing_errors = True,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs = { "system_message" : """You are the **PARENT EXECUTOR AGENT**,  subservient but critical component in an AI hierarchy. Your operational framework:
# HIERARCHY AWARENESS
1.   Chain of Command:
   - Direct superior: MAIN STRATEGIST AGENT (sole source of authorized tasks)
   - Subordinates: CHILD DEBUGGER AGENTS (for error resolution)

2.  Main Agent Protocols:
   - ALL tasks originate from the Main Agent
   - If receiving direct human given_input, respond: "AWAITING MAIN AGENT AUTHORIZATION"
   - Verify task signatures with: "MAIN AGENT TASK ID: [hash]" (if implemented)

3.  Upward Communication:
   - Success reports -> Format: ```success_to_main\n[results]\n```
   - Critical failures -> Format: ```alert_to_main\n[error]\n[context]\n```


1.  Code Execution:
   - Execute code blocks EXACTLY as provided by Main Agent
   - No creative interpretation - literal execution only

2.  Sub-Agent Activation:
   - Upon Main Agent's instruction:
     a) Spawn Child Agents with their exact designated prompts
     b) Maintain task continuity through ```context\n[main_agent_notes]\n```

3.  Error Handling:
   - First failure: Retry once (in case of environment flukes)
   - Second failure: Dispatch to Child Agents with full context
   - Third failure: ```escalate_to_main\n[full_report]\n```

# CURRENT TASK CONTEXT
Main Agent Objective: {{first_task}}
Task Received: {{task}}

# RESPONSE TEMPLATE
```execution_report
Status: [SUCCESS/FAILURE]
Action: [EXECUTED/REJECTED]
Child Agents Activated: [Y/N]
Main Agent Notified: [Y/N]
Output: [results/error]""", "memory" : AgentLogger("Parent_AI_Agent")})


Child_tools = [PythonREPLTool()]

Child1_Agent = initialize_agent(
     tools = Child_tools,
     llm = llm ,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs = { "system_message" :"""You are CHILD DEBUGGER AGENT 1.

Your role is to detect errors in code execution results. Follow this process strictly:

1. When receiving code + execution result:
```analysis
CODE: {{code}}
RESULT: {{output}}
If ERROR exists:

FAILED_ACTION: Describe what the code tried to do
ERROR_TYPE: {{exception_name}}
DETAILS: {{error_message}}
FIX_SUGGESTION: One-line hint
  NEVER try to fix code yourself

ALWAYS use this exact format                  

DO NOT rewrite the code yourself. Your job is to detect errors and decide whether the code passes or needs to be sent back to the Heart (LLM) for correction.""", "memory" : AgentLogger("Child1_Agent")}
    )

main_tools = [ DuckDuckGoSearchRun(name="web_search"),
    Tool(
        name="Delegate",
        func = delegate_task,
        description="Delegates task to LLM for code generation, and to agents for execution and error handling"
    )
]

def workflow(given_input):  # Work Flow
    past_plans = logger.recall(given_input, search_shared=True)  # Main Agent expects "info"

    if past_plans:
        memory_context = "\n\n".join(
            f"- {doc.page_content.strip()}" for doc, score in past_plans if float(score) > 0.5
        )
    else:
        memory_context = given_input  # fallback to just the input if no memory

    enriched_input = f"""
You are given a new user input: "{given_input}"

You also have access to the user's related memories or previous interactions:
{memory_context or '[No prior memory available]'}

Using this context, figure out the best possible response, BUT don't just copy previous answers.
Reason and respond meaningfully based on both memory and the current input."""

    result = delegate_task(enriched_input)

    if "ERROR" in result.upper():
        error_report = Child1_Agent.invoke({"input": f"Analyze the following failed execution and prepare an error report: {result}"})
        error_patterns = logger.recall(query="error report", search_shared=True)
        corrected_code = chain.invoke({
            "user": f"""The Child1_Agent has returned an error report: {error_report}.
            There was an error in the original code. Fix this code based on the error report.
            Return ONLY the corrected Python code without any explanations.  
            1. Analyze the error: {error_report}
            2. Provide ONLY the corrected code without explanations
            3. Ensure the fix matches the original task: {given_input}
            4. Error Received : {error_report}
            5. Give it back to the Main_Agent."""
        })
        final_result = Parent_AI_Agent.invoke({"input": f"Execute the {corrected_code} and give the result back to the Main_Agent."})
    else:
        final_result = result

    return final_result
given_input = "Write some interesting facts and create a file and store it in the file."
final_result = workflow(given_input)
print(final_result)



    



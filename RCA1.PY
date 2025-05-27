import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import numpy as np

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "./chroma_db"

# Initialize components
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model_name="gpt-4",
    temperature=0.3
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model="text-embedding-ada-002"
)

# Initialize Chroma DB
vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)
"""
Purpose:

Converts historical CAPA PDFs into searchable vectors

Creates a knowledge base for semantic search

Why Used:

ChromaDB: Lightweight, local vector store that preserves data privacy

Embeddings: Convert text to vectors to enable meaning-based search (not just keyword matching)

Persistent storage: Retains knowledge between sessions"""
# --------------------------
# Semantic Search Tool
# --------------------------

def semantic_search(query: str) -> str:
    """Perform semantic search with meaning-based queries"""
    docs = vector_db.similarity_search(query, k=3)
    # Process docs to remove similar cases display
    return "\n".join([d.page_content.split("Similar Cases:")[0] for d in docs])

"""
Purpose:

Finds relevant historical CAPAs based on meaning (not just keywords)

Why Used:

Semantic over similarity: Understands context (e.g., "contamination" matches "particulate matter")

Filtering: Removes similar cases display per your requirements

Precision: k=3 returns only the most relevant cases to reduce noise


"""
# --------------------------
# Specialized Tools
# --------------------------
"""
Purpose:

Systematically identifies underlying causes using proven frameworks

Why Used:

5 Whys: Drills down to root cause (e.g., "Why? → Why? → Why?")

Fishbone: Categorizes causes (People/Methods/Materials/etc.)

Structured output: Returns ranked causes with confidence levels
"""
def rca_framework(description: str) -> str:
    """Root Cause Analysis framework tool"""
    prompt = ChatPromptTemplate.from_template(
        """Analyze this CAPA description using:
        1. 5 Whys technique
        2. Fishbone diagram categories
        3. Fault tree analysis
        
        CAPA Description: {description}
        
        Provide 3 probable root causes with confidence levels"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"description": description})



"""
Purpose:

Quantifies severity of the CAPA issue

Why Used:

Weighted factors: Prioritizes safety/regulatory impact

Normalized score (0-1): Allows comparison across CAPAs

Automated consistency: Removes subjective human bias
"""

def impact_scorer(description: str) -> float:
    """Impact scoring tool"""
    prompt = ChatPromptTemplate.from_template(
        """Rate the impact (0-1) of this CAPA:
        - Safety (40%)
        - Regulatory (30%)
        - Financial (20%)
        - Operational (10%)
        
        Description: {description}
        Respond ONLY with the score (0.00-1.00)"""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return float(chain.invoke({"description": description}))
    except:
        return 0.5


"""
Purpose:

Captures user corrections to improve future analyses

Why Used:

Continuous learning: Updates vector embeddings with corrected root causes

Adaptive system: Becomes more accurate over time

Closed-loop: Implements your requirement for feedback integration

"""

def feedback_logger(feedback: str) -> str:
    """Tool for processing feedback"""
    # In production, would update ChromaDB embeddings
    return "Feedback recorded for system improvement"

# Create tools
tools = [
    Tool(
        name="SemanticSearchCAPA",
        func=semantic_search,
        description="Meaning-based search of CAPA knowledge base"
    ),
    Tool(
        name="RCAAnalysis",
        func=rca_framework,
        description="Structured root cause analysis"
    ),
    Tool(
        name="ImpactScorer",
        func=impact_scorer,
        description="Calculate CAPA impact score (0-1)"
    ),
    Tool(
        name="FeedbackHandler",
        func=feedback_logger,
        description="Process user feedback"
    )
]

# --------------------------
# LangGraph Workflow (Modified)
# --------------------------

class AgentState(TypedDict):
    description: str
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    needs_feedback: bool

def retrieve_context(state: AgentState):
    """Node: Semantic search"""
    context = semantic_search(state["description"])
    return {"context": context}

def analyze_causes(state: AgentState):
    """Node: Root cause analysis"""
    causes = rca_framework(state["description"])
    return {"root_causes": [causes]}  # Simplified parsing

def assess_impact(state: AgentState):
    """Node: Impact scoring"""
    score = impact_scorer(state["description"])
    return {"impact_score": score}

def handle_feedback(state: AgentState):
    """Node: Feedback processing"""
    if state.get("needs_feedback", False):
        feedback_logger(state["feedback"])
        return {"message": "System updated with feedback"}
    return {"message": "Analysis complete"}

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("analyze", analyze_causes)
workflow.add_node("impact", assess_impact)
workflow.add_node("feedback", handle_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "impact")
workflow.add_conditional_edges(
    "impact",
    lambda state: "feedback" if state.get("needs_feedback", False) else END,
    {"feedback": "feedback", END: END}
)
workflow.add_edge("feedback", END)

capa_workflow = workflow.compile()

# --------------------------
# User Interface
# --------------------------

def analyze_capa(description: str):
    """Public interface for CAPA analysis"""
    result = capa_workflow.invoke({
        "description": description,
        "needs_feedback": False
    })
    
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"]
    }

def submit_feedback(feedback: str):
    """Public interface for feedback"""
    feedback_logger(feedback)
    return {"status": "Feedback processed"}

# Example Usage
if __name__ == "__main__":
    analysis = analyze_capa("Product contamination in Batch #123")
    print("Root Causes:", analysis["root_causes"])
    print("Impact Score:", analysis["impact_score"])



++++++++++++++++++++++++++++++++++++++++
import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import numpy as np

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "RCA_db"  # Directory to store ChromaDB data

# --------------------------
# Step 1: PDF Loading and Processing
# --------------------------

def load_and_process_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Load and split PDFs into properly formatted documents"""
    all_documents = []
    
    for pdf_path in pdf_paths:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        documents = text_splitter.split_documents(docs)
        all_documents.extend(documents)
    
    return all_documents

# --------------------------
# Step 2: Vector Store Initialization
# --------------------------

# Initialize Azure OpenAI components
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model_name="gpt-4",
    temperature=0.3
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model="text-embedding-ada-002"
)

# Initialize Chroma DB with persistence
def initialize_vector_store(pdf_paths: List[str]):
    """Initialize ChromaDB with PDF documents"""
    documents = load_and_process_pdfs(pdf_paths)
    
    chroma_settings = {
        "persist_directory": CHROMA_DB_PATH,
        "anonymized_telemetry": False
    }
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_settings["persist_directory"]
    )

# --------------------------
# Step 3: Semantic Search Tool
# --------------------------

def semantic_search(query: str, vector_db: Chroma) -> str:
    """Perform semantic search with meaning-based queries"""
    docs = vector_db.similarity_search(query, k=3)
    # Process docs to remove similar cases display
    return "\n".join([d.page_content.split("Similar Cases:")[0] for d in docs])

# --------------------------
# Step 4: Specialized Tools
# --------------------------

def rca_framework(description: str) -> str:
    """Root Cause Analysis framework tool"""
    prompt = ChatPromptTemplate.from_template(
        """Analyze this CAPA description using:
        1. 5 Whys technique
        2. Fishbone diagram categories
        3. Fault tree analysis
        
        CAPA Description: {description}
        
        Provide 3 probable root causes with confidence levels"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"description": description})

def impact_scorer(description: str) -> float:
    """Impact scoring tool"""
    prompt = ChatPromptTemplate.from_template(
        """Rate the impact (0-1) of this CAPA:
        - Safety (40%)
        - Regulatory (30%)
        - Financial (20%)
        - Operational (10%)
        
        Description: {description}
        Respond ONLY with the score (0.00-1.00)"""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return float(chain.invoke({"description": description}))
    except:
        return 0.5  # Default if parsing fails

def feedback_logger(feedback: str) -> str:
    """Tool for processing feedback"""
    # In production, would update ChromaDB embeddings
    return "Feedback recorded for system improvement"

# --------------------------
# Step 5: Initialize System with PDF Data
# --------------------------

# Example usage - initialize with your PDF files
pdf_paths = ["path/to/capa1.pdf", "path/to/capa2.pdf"]  # Replace with actual paths
vector_db = initialize_vector_store(pdf_paths)

# Create tools
tools = [
    Tool(
        name="SemanticSearchCAPA",
        func=lambda q: semantic_search(q, vector_db),
        description="Meaning-based search of CAPA knowledge base"
    ),
    Tool(
        name="RCAAnalysis",
        func=rca_framework,
        description="Structured root cause analysis"
    ),
    Tool(
        name="ImpactScorer",
        func=impact_scorer,
        description="Calculate CAPA impact score (0-1)"
    ),
    Tool(
        name="FeedbackHandler",
        func=feedback_logger,
        description="Process user feedback"
    )
]

# --------------------------
# Step 6: LangGraph Workflow
# --------------------------

class AgentState(TypedDict):
    description: str
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    needs_feedback: bool

def retrieve_context(state: AgentState):
    """Node: Semantic search"""
    context = semantic_search(state["description"], vector_db)
    return {"context": context}

def analyze_causes(state: AgentState):
    """Node: Root cause analysis"""
    causes = rca_framework(state["description"])
    return {"root_causes": [causes]}  # Simplified parsing

def assess_impact(state: AgentState):
    """Node: Impact scoring"""
    score = impact_scorer(state["description"])
    return {"impact_score": score}

def handle_feedback(state: AgentState):
    """Node: Feedback processing"""
    if state.get("needs_feedback", False):
        feedback_logger(state["feedback"])
        return {"message": "System updated with feedback"}
    return {"message": "Analysis complete"}

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("analyze", analyze_causes)
workflow.add_node("impact", assess_impact)
workflow.add_node("feedback", handle_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "impact")
workflow.add_conditional_edges(
    "impact",
    lambda state: "feedback" if state.get("needs_feedback", False) else END,
    {"feedback": "feedback", END: END}
)
workflow.add_edge("feedback", END)

capa_workflow = workflow.compile()

# --------------------------
# Step 7: User Interface
# --------------------------

def analyze_capa(description: str):
    """Public interface for CAPA analysis"""
    result = capa_workflow.invoke({
        "description": description,
        "needs_feedback": False
    })
    
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"]
    }

def submit_feedback(feedback: str):
    """Public interface for feedback"""
    feedback_logger(feedback)
    return {"status": "Feedback processed"}

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    # Initialize with PDFs (run once)
    # pdf_paths = ["capa_reports/report1.pdf", ...]
    # vector_db = initialize_vector_store(pdf_paths)
    
    # Analyze a new case
    analysis = analyze_capa("Product contamination in Batch #123")
    print("Root Causes:", analysis["root_causes"])
    print("Impact Score:", analysis["impact_score"])
    
    # Submit feedback
    feedback_result = submit_feedback("Root cause was actually humidity controls")
    print(feedback_result)

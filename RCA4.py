import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import Tool
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
import numpy as np
import shutil

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "./chroma_db"
PDF_SOURCE_DIR = "./capa_pdfs"

# --------------------------
# PDF Preprocessing Pipeline
# --------------------------

def initialize_vector_store(pdf_dir: str, recreate: bool = True) -> Chroma:
    """Initialize ChromaDB with CAPA documents from PDFs"""
    if recreate and os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_dir, filename))
                pdf_docs = loader.load()
                chunks = text_splitter.split_documents(pdf_docs)
                documents.extend(chunks)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    embeddings = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model="text-embedding-ada-002"
    )
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

# Initialize vector store
vector_db = initialize_vector_store(PDF_SOURCE_DIR)

# --------------------------
# Analysis Tools
# --------------------------

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model_name="gpt-4",
    temperature=0.3
)

def semantic_search(query: str) -> str:
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

"""
1) 5 Whys: Drills down to root cause (e.g., "Why? → Why? → Why?")

2) Fishbone: Categorizes causes (People/Methods/Materials/etc.)

3) Fault Tree Analysis (FTA) Explanation
What it is:

A top-down, deductive failure analysis method

Uses Boolean logic to model causal relationships

Visualized as an inverted tree with:

Top event (failure) as the root

Intermediate events as branches

Basic causes as leaves

How it works :


Fault tree analysis  # << This triggers the LLM to:
   a. Identify the top-level failure (from CAPA description)
   b. Decompose into contributing factors
   c. Apply AND/OR gates to determine combinations
   d. Trace paths to fundamental causes

Confidence Level Calculation
How it's determined:

The LLM evaluates each root cause using:

Evidence strength: How directly the cause explains the effect

Path certainty: Clearness of the fault tree path

Historical correlation: Frequency in similar CAPAs

Scoring approach:

Example
# Hypothetical LLM reasoning process
causes = [
    {
        "cause": "Inadequate staff training",
        "evidence": ["No training records", "Similar past incidents"],
        "confidence": 0.85  # 85%
    },
    {
        "cause": "Equipment calibration drift",
        "evidence": ["Last calibration 6mo ago"],
        "confidence": 0.65
    }
]   

   """
def rca_framework(description: str) -> str:
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

def feedback_logger(feedback: str) -> str:
    return "Feedback recorded for system improvement"

tools = [
    Tool(
        name="SemanticSearchCAPA",
        func=semantic_search,
        description="Search CAPA knowledge base"
    ),
    Tool(
        name="RCAAnalysis",
        func=rca_framework,
        description="Perform root cause analysis"
    ),
    Tool(
        name="ImpactScorer",
        func=impact_scorer,
        description="Calculate impact score"
    ),
    Tool(
        name="FeedbackHandler",
        func=feedback_logger,
        description="Process user feedback"
    )
]

# --------------------------
# Workflow Definition (Fixed)
# --------------------------

class AgentState(TypedDict):
    description: str
    context: Optional[str]
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    has_feedback: bool

def retrieve_context(state: AgentState):
    context = semantic_search(state["description"])
    return {"context": context}

def analyze_causes(state: AgentState):
    causes = rca_framework(state["description"])
    return {"root_causes": causes.split("\n") if causes else []}

def assess_impact(state: AgentState):
    score = impact_scorer(state["description"])
    return {"impact_score": score}

def handle_feedback(state: AgentState):
    if state.get("has_feedback", False):
        feedback_logger(state["feedback"])
    return {"message": "Feedback processed" if state.get("has_feedback") else "No feedback"}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("analyze", analyze_causes)
workflow.add_node("assess", assess_impact)
workflow.add_node("handle_feedback", handle_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "assess")
workflow.add_conditional_edges(
    "assess",
    lambda state: "handle_feedback" if state.get("has_feedback", False) else END,
    {"handle_feedback": "handle_feedback", END: END}
)
workflow.add_edge("handle_feedback", END)

capa_workflow = workflow.compile()

# --------------------------
# User Interface (Fixed)
# --------------------------

def analyze_capa(description: str):
    result = capa_workflow.invoke({
        "description": description,
        "has_feedback": False
    })
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"]
    }

def submit_feedback(feedback: str):
    result = capa_workflow.invoke({
        "description": "",
        "feedback": feedback,
        "has_feedback": True
    })
    return {"status": result["message"]}

# Example Usage
if __name__ == "__main__":
    # First run: Initialize with PDFs
    # vector_db = initialize_vector_store(PDF_SOURCE_DIR, recreate=True)
    
    analysis = analyze_capa("Product contamination in Batch #123")
    print("Root Causes:")
    for cause in analysis["root_causes"]:
        print(f"- {cause}")
    print(f"Impact Score: {analysis['impact_score']:.2f}")"

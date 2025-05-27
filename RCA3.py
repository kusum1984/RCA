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
import shutil

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "./chroma_db"
PDF_SOURCE_DIR = "./capa_pdfs"  # Directory containing your CAPA PDFs

# --------------------------
# PDF Preprocessing Pipeline (NEW)
# --------------------------

def initialize_vector_store(pdf_dir: str, recreate: bool = True) -> Chroma:
    """Initialize ChromaDB with CAPA documents from PDFs"""
    if recreate and os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Load and split all PDFs
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
    
    # Create and persist ChromaDB
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

# Initialize vector store with PDFs (ONE-TIME SETUP)
vector_db = initialize_vector_store(PDF_SOURCE_DIR)

# --------------------------
# Existing Workflow (UNCHANGED)
# --------------------------

# Initialize components
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model_name="gpt-4",
    temperature=0.3
)

def semantic_search(query: str) -> str:
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([d.page_content.split("Similar Cases:")[0] for d in docs])

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

class AgentState(TypedDict):
    description: str
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    needs_feedback: bool

def retrieve_context(state: AgentState):
    context = semantic_search(state["description"])
    return {"context": context}

def analyze_causes(state: AgentState):
    causes = rca_framework(state["description"])
    return {"root_causes": [causes]}

def assess_impact(state: AgentState):
    score = impact_scorer(state["description"])
    return {"impact_score": score}

def handle_feedback(state: AgentState):
    if state.get("needs_feedback", False):
        feedback_logger(state["feedback"])
        return {"message": "System updated with feedback"}
    return {"message": "Analysis complete"}

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
# User Interface (UNCHANGED)
# --------------------------

def analyze_capa(description: str):
    result = capa_workflow.invoke({
        "description": description,
        "needs_feedback": False
    })
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"]
    }

def submit_feedback(feedback: str):
    feedback_logger(feedback)
    return {"status": "Feedback processed"}

# Example Usage
if __name__ == "__main__":
    # First run: Initialize with PDFs (uncomment to rebuild)
    # vector_db = initialize_vector_store(PDF_SOURCE_DIR, recreate=True)
    
    analysis = analyze_capa("Product contamination in Batch #123")
    print("Root Causes:", analysis["root_causes"])
    print("Impact Score:", analysis["impact_score"])

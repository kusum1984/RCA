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
import json
from datetime import datetime
from pathlib import Path

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "./chroma_db"
PDF_SOURCE_DIR = "./capa_pdfs"
FEEDBACK_DB = "./feedback_logs.json"

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

def feedback_logger(feedback: str, context: str = None) -> str:
    """Enhanced feedback logger with persistent storage"""
    Path(FEEDBACK_DB).parent.mkdir(exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback,
        "context": context,
        "status": "new"
    }
    
    try:
        with open(FEEDBACK_DB, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return f"Feedback logged at {entry['timestamp']}"
    except Exception as e:
        return f"Error logging feedback: {str(e)}"

def get_feedback_reports() -> List[Dict]:
    """Retrieve all feedback entries"""
    if not os.path.exists(FEEDBACK_DB):
        return []
    
    with open(FEEDBACK_DB, "r") as f:
        return [json.loads(line) for line in f.readlines()]

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
        func=lambda fb: feedback_logger(fb["feedback"], fb.get("context")),
        description="Process user feedback with context logging"
    )
]

# --------------------------
# Workflow Definition
# --------------------------

class AgentState(TypedDict):
    description: str
    context: Optional[str]
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    has_feedback: bool
    iteration: Optional[int]

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
        context = f"""
        Analysis Context (Iteration {state.get('iteration', 1)}):
        - Description: {state['description']}
        - Root Causes: {state['root_causes']}
        - Impact Score: {state['impact_score']}
        """
        return {
            "message": feedback_logger(state["feedback"], context=context)
        }
    return {"message": "No feedback to process"}

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
workflow.set_finish_point(END)
capa_workflow = workflow.compile()

# --------------------------
# Enhanced User Interface
# --------------------------

def analyze_with_feedback(initial_description: str, max_iterations: int = 3):
    """Run CAPA analysis with feedback-driven iteration"""
    results = []
    current_description = initial_description
    feedback = None
    
    for iteration in range(1, max_iterations + 1):
        # Run analysis
        result = capa_workflow.invoke({
            "description": current_description,
            "feedback": feedback,
            "has_feedback": (feedback is not None),
            "iteration": iteration
        })
        results.append(result)
        
        # Show results
        print(f"\n=== Iteration {iteration} Results ===")
        print("Root Causes:")
        for cause in result["root_causes"]:
            print(f"- {cause}")
        print(f"Impact Score: {result['impact_score']:.2f}")
        
        # Get user feedback
        user_input = input("\nProvide feedback to improve analysis (or press Enter to accept): ").strip()
        if not user_input:
            break
            
        # Prepare next iteration
        feedback = user_input
        current_description = f"""{initial_description}
        
        === User Feedback ===
        {feedback}
        
        === Previous Analysis (Iteration {iteration}) ===
        Root Causes:
        {chr(10).join(f"- {cause}" for cause in result["root_causes"])}
        Impact Score: {result['impact_score']:.2f}
        """
    
    return {
        "final_results": results[-1],
        "all_iterations": results,
        "feedback_used": feedback is not None
    }

def analyze_capa(description: str, feedback: Optional[str] = None):
    """Original single-pass analysis with optional feedback"""
    result = capa_workflow.invoke({
        "description": description,
        "feedback": feedback,
        "has_feedback": feedback is not None
    })
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"],
        "feedback_status": result.get("message", "")
    }

def submit_feedback(feedback: str):
    """Standalone feedback submission"""
    result = capa_workflow.invoke({
        "description": "",
        "feedback": feedback,
        "has_feedback": True
    })
    return {"status": result["message"]}

def view_feedback():
    """View all feedback entries"""
    return get_feedback_reports()

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    print("CAPA Analysis System")
    print("1. Single analysis\n2. Interactive analysis with feedback")
    choice = input("Select mode (1/2): ")
    
    description = input("Enter CAPA description: ")
    
    if choice == "1":
        # Original single-pass mode
        result = analyze_capa(description)
        print("\nResults:")
        for cause in result["root_causes"]:
            print(f"- {cause}")
        print(f"Impact Score: {result['impact_score']:.2f}")
    else:
        # Enhanced feedback-driven mode
        results = analyze_with_feedback(description)
        print("\nFinal Results:")
        for cause in results["final_results"]["root_causes"]:
            print(f"- {cause}")
        print(f"Impact Score: {results['final_results']['impact_score']:.2f}")
        print(f"Total iterations: {len(results['all_iterations'])}")
    
    # Feedback history available via view_feedback()

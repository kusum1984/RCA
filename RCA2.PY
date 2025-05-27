import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import Tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import shutil

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "capa_vector_db"
PDF_DIRECTORY = "capa_pdfs"  # Directory containing your CAPA PDFs

# --------------------------
# Step 1: PDF Processing Pipeline
# --------------------------

class CAPADocument(BaseModel):
    content: str
    metadata: dict = {}

def load_and_chunk_pdfs(pdf_dir: str) -> List[CAPADocument]:
    """Load and process all PDFs in directory"""
    documents = []
    
    # Create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    
    # Process each PDF file
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                loader = PyPDFLoader(filepath)
                pdf_docs = loader.load()
                
                # Split and add metadata
                for doc in pdf_docs:
                    chunks = text_splitter.split_documents([doc])
                    for chunk in chunks:
                        documents.append(CAPADocument(
                            content=chunk.page_content,
                            metadata={
                                "source": filename,
                                "page": chunk.metadata.get("page", 0)
                            }
                        ))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return documents

# --------------------------
# Step 2: Vector Store Setup
# --------------------------

def initialize_vector_store(pdf_dir: str, recreate: bool = False) -> Chroma:
    """Initialize ChromaDB with CAPA documents"""
    if recreate and os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model="text-embedding-ada-002"
    )
    
    # Load and process documents
    capa_docs = load_and_chunk_pdfs(pdf_dir)
    
    # Create ChromaDB instance
    return Chroma.from_documents(
        documents=[Document(page_content=doc.content, metadata=doc.metadata) for doc in capa_docs],
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

# --------------------------
# Step 3: Analysis Tools
# --------------------------

def create_analysis_tools(vector_db: Chroma):
    """Create tools for CAPA analysis"""
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model_name="gpt-4",
        temperature=0.3
    )
    
    # Semantic Search Tool
    def semantic_search(query: str) -> str:
        docs = vector_db.similarity_search(query, k=3)
        return "\n---\n".join(d.page_content for d in docs)
    
    # RCA Analysis Tool
    def rca_analysis(description: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            """Perform Root Cause Analysis for:
            {description}
            
            Use:
            1. 5 Whys methodology
            2. Fishbone diagram categories
            3. Fault tree analysis
            
            Return top 3 root causes with confidence scores"""
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"description": description})
    
    # Impact Scoring Tool
    def impact_score(description: str) -> float:
        prompt = ChatPromptTemplate.from_template(
            """Calculate impact score (0-1) for:
            {description}
            
            Consider:
            - Safety (40%)
            - Regulatory (30%)
            - Financial (20%)
            - Operational (10%)
            
            Return ONLY the score"""
        )
        chain = prompt | llm | StrOutputParser()
        try:
            return float(chain.invoke({"description": description}))
        except:
            return 0.5
    
    return [
        Tool(
            name="semantic_search",
            func=semantic_search,
            description="Search CAPA knowledge base using semantic similarity"
        ),
        Tool(
            name="rca_analysis",
            func=rca_analysis,
            description="Perform root cause analysis using structured methods"
        ),
        Tool(
            name="impact_scorer",
            func=impact_score,
            description="Calculate impact score (0-1) for CAPA issues"
        )
    ]

# --------------------------
# Step 4: Workflow Setup
# --------------------------

def create_workflow(tools: List[Tool]):
    """Create CAPA analysis workflow"""
    
    class AnalysisState(TypedDict):
        description: str
        context: str
        root_causes: List[str]
        impact_score: float
    
    # Define nodes
    def retrieve(state: AnalysisState):
        return {"context": tools[0].func(state["description"])}
    
    def analyze(state: AnalysisState):
        return {"root_causes": [tools[1].func(state["description"])]}
    
    def score(state: AnalysisState):
        return {"impact_score": tools[2].func(state["description"])}
    
    # Build workflow
    workflow = StateGraph(AnalysisState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("analyze", analyze)
    workflow.add_node("score", score)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "score")
    workflow.add_edge("score", END)
    
    return workflow.compile()

# --------------------------
# Step 5: System Initialization
# --------------------------

class CAPAAnalyzer:
    def __init__(self, pdf_directory: str):
        self.vector_db = initialize_vector_store(pdf_directory, recreate=True)
        self.tools = create_analysis_tools(self.vector_db)
        self.workflow = create_workflow(self.tools)
    
    def analyze(self, description: str) -> dict:
        """Analyze a CAPA description"""
        result = self.workflow.invoke({"description": description})
        return {
            "root_causes": result["root_causes"],
            "impact_score": result["impact_score"],
            "context": result["context"]
        }

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    # Initialize system with PDFs
    analyzer = CAPAAnalyzer(PDF_DIRECTORY)
    
    # Analyze a new CAPA case
    analysis = analyzer.analyze(
        "Batch 123 failed sterility testing due to particulate contamination"
    )
    
    print("Analysis Results:")
    print(f"Root Causes: {analysis['root_causes']}")
    print(f"Impact Score: {analysis['impact_score']:.2f}")

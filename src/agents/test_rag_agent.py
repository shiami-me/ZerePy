import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_agent import RAGAgent, EnhancedKnowledgeGraphTool, EnhancedRAGQueryTool

load_dotenv()

def initialize_rag_agent():
    """Initialize the RAG agent with Google's Gemini model."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            api_key=os.getenv("GEMINI_API_KEY")
        )
        prompt = """
        You are an advanced RAG agent capable of processing documents, building knowledge graphs, 
        and answering complex queries using hybrid search (vector + graph-based approaches).
        
        For document processing:
        1. Extract key entities and their relationships
        2. Build a comprehensive knowledge graph
        3. Create vector embeddings for efficient retrieval
        
        For query answering:
        1. Use both vector similarity and graph traversal
        2. Combine results for comprehensive answers
        3. Provide context and sources when possible
        """
        return RAGAgent(llm=llm, name="rag_agent", prompt=prompt)
    except Exception as e:
        print(f"Error initializing RAG agent: {e}")
        raise

def test_document_processing(file_path):
    """Test the document processing functionality."""
    print(f"\nProcessing document: {file_path}")
    try:
        kg_tool = EnhancedKnowledgeGraphTool()
        result = kg_tool._run(file_path)
        print("\nDocument Processing Result:")
        print(result)
        return True
    except Exception as e:
        print(f"Error during document processing: {e}")
        return False

def test_rag_query(query, require_graph=True):
    """Test the RAG query functionality."""
    print(f"\nQuerying RAG system: {query}")
    try:
        query_tool = EnhancedRAGQueryTool()
        
        # Check if any knowledge graphs exist
        graph_dir = query_tool.graph_manager.storage_dir
        if require_graph and not any(f.endswith('.graphml') for f in os.listdir(graph_dir)):
            print("Warning: No knowledge graphs found. Please process a document first.")
            return False
            
        result = query_tool._run(query)
        print("\nQuery Result:")
        print(result)
        return True
    except Exception as e:
        print(f"Error during query: {e}")
        return False

def test_agent_interaction(rag_agent, query):
    """Test the full RAG agent interaction."""
    print(f"\nInvoking RAG agent with query: {query}")
    try:
        result = rag_agent.invoke(query)
        print("\nAgent Response:")
        print(result)
        return True
    except Exception as e:
        print(f"Error during agent interaction: {e}")
        return False

def ensure_directories():
    """Ensure required directories exist."""
    directories = ['knowledge_graphs', 'temp', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    # Ensure required directories exist
    ensure_directories()
    
    # Initialize the RAG agent
    try:
        rag_agent = initialize_rag_agent()
    except Exception as e:
        print(f"Failed to initialize RAG agent: {e}")
        return

    # Test document processing
    sample_file_path = "Chirag Khandelwal Resume.pdf"  # Update with your test document
    if os.path.exists(sample_file_path):
        success = test_document_processing(sample_file_path)
        if not success:
            print("Document processing failed. Subsequent tests may not work as expected.")
    else:
        print(f"Warning: Test file not found at {sample_file_path}")
        print("Please provide a valid document path to test document processing.")

    # Test queries
    test_queries = [
        "What are the key concepts in the document?",
        "Explain the relationships between the main entities.",
        "Summarize the document content."
    ]
    
    for query in test_queries:
        test_rag_query(query, require_graph=True)

    # Test full agent interaction
    agent_query = "Provide a summary and key insights from the document."
    test_agent_interaction(rag_agent, agent_query)

if __name__ == "__main__":
    main()
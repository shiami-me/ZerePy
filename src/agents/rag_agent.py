from langchain_core.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent
from ..tools.rag_tools import GraphRAG, DocumentProcessor, KnowledgeGraph, QueryEngine, Visualizer
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

class State(MessagesState):
    next: str
class RAGQueryResult(BaseModel):
    """Result of a RAG query operation"""
    answer: str = Field(description="The answer retrieved from the knowledge graph")
    traversal_info: str = Field(description="Information about how the knowledge graph was traversed")

class RAGAgent:
    def __init__(self, llm, name: str, next: str):
        """Initialize the RAG Agent with LLM and graph components"""
        self._name = name
        self.next = next
        self.llm = llm
        
        # Define tools
        self.tools = [
            self.load_documents,
            self.initialize_graph_rag,
            self.query_knowledge_graph, 
            self.visualize_traversal
        ]
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant with access to a Graph-based RAG system. 
            You can help users load documents, build knowledge graphs, and answer questions based on the loaded documents.
            
            Follow these steps when helping users:
            1. First load documents using the load_documents tool
            2. Initialize the knowledge graph using initialize_graph_rag
            3. Answer queries using query_knowledge_graph
            4. Visualize the graph traversal path when requested
            
            Think step by step about the best way to use these tools to help the user.
            """),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create vector store utils
        vector_store = VectorStoreUtils(tools=self.tools)
        
        # Create agent using the Agent class with proper parameters
        self.agent = Agent(
            tools=self.tools,
            vector_store=vector_store,
            llm=llm,
            prompt=self.prompt
        )._create_conversation_graph()

        # Global variables to maintain state
        self.documents = None
        self.graph_rag = None
        self.traversal_path = None
        self.filtered_content = None

    @tool
    def load_documents(self, file_path: str) -> str:
        """Load documents from the specified PDF file path"""
        try:
            loader = PyPDFLoader(file_path)
            self.documents = loader.load()
            # For testing purposes, limit to first 20 pages or chunks
            self.documents = self.documents[:20]
            
            return f"Successfully loaded {len(self.documents)} document chunks from {file_path}"
        except Exception as e:
            return f"Error loading documents: {str(e)}"
    
    @tool
    def initialize_graph_rag(self) -> str:
        """Initialize the GraphRAG system with the loaded documents"""
        try:
            if self.documents is None:
                return "Please load documents first using the load_documents tool"
            
            self.graph_rag = GraphRAG(self.documents)
            
            return "Successfully initialized GraphRAG system with the loaded documents"
        except Exception as e:
            return f"Error initializing GraphRAG: {str(e)}"
    
    @tool
    def query_knowledge_graph(self, query: str) -> RAGQueryResult:
        """Query the knowledge graph with the given query and return the answer"""
        try:
            if self.graph_rag is None:
                return RAGQueryResult(
                    answer="Please initialize the GraphRAG system first",
                    traversal_info="No traversal occurred as GraphRAG is not initialized."
                )
            
            # Get response from query engine
            response, traversal_path, filtered_content = self.graph_rag.query_engine.query(query)
            
            # Store traversal path and filtered content for visualization
            self.traversal_path = traversal_path
            self.filtered_content = filtered_content
            
            # Create traversal info summary
            traversal_info = f"The query traversed {len(traversal_path)} nodes in the knowledge graph."
            if traversal_path:
                traversal_info += f" Starting from node {traversal_path[0]} and ending at node {traversal_path[-1]}."
            
            return RAGQueryResult(
                answer=response,
                traversal_info=traversal_info
            )
        except Exception as e:
            return RAGQueryResult(
                answer=f"Error querying knowledge graph: {str(e)}",
                traversal_info="No traversal occurred due to an error."
            )
    
    @tool
    def visualize_traversal(self, show_visualization: bool) -> str:
        """Visualize the latest traversal path in the knowledge graph"""
        try:
            if not show_visualization:
                return "Visualization skipped as requested"
                
            if not self.traversal_path or not self.graph_rag:
                return "No traversal path available to visualize. Please run a query first."
                
            # Initialize visualizer
            visualizer = Visualizer()
            
            # Generate visualization
            visualizer.visualize_traversal(
                self.graph_rag.knowledge_graph.graph,
                self.traversal_path
            )
            
            # Print filtered content details
            visualizer.print_filtered_content(
                self.traversal_path,
                self.filtered_content
            )
            
            return "Visualization of the traversal path has been generated"
        except Exception as e:
            return f"Error generating visualization: {str(e)}"
    
    def node(self, state: State):
        """Process the current state and determine the next action"""
        # Update instance variables from state if they exist in state
        if hasattr(state, 'documents') and state.documents is not None:
            self.documents = state.documents
        if hasattr(state, 'graph_rag') and state.graph_rag is not None:
            self.graph_rag = state.graph_rag
        if hasattr(state, 'traversal_path') and state.traversal_path is not None:
            self.traversal_path = state.traversal_path
        if hasattr(state, 'filtered_content') and state.filtered_content is not None:
            self.filtered_content = state.filtered_content
        
        # Invoke the agent
        result = self.agent.invoke(state)
        
        # Update state with latest instance variables
        updates = {}
        updates["graph_rag"] = self.graph_rag
        updates["documents"] = self.documents
        updates["traversal_path"] = self.traversal_path
        updates["filtered_content"] = self.filtered_content
        
        # Add the agent's response message
        updates["messages"] = [AIMessage(content=result["messages"][-1].content, name=self._name)]
        
        # Return command with updates and next node
        return Command(
            update=updates,
            goto=self.next
        )
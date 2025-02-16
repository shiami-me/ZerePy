from typing import Type, Optional, List, Tuple
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
import networkx as nx
import spacy
from spacy.tokens import Doc
import os
import json
from datetime import datetime

# Core components
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize spaCy with English medium model
nlp = spacy.load("en_core_web_md")

class RAGInput(BaseModel):
    query: str = Field(description="User query for RAG retrieval")
    file_path: Optional[str] = Field(description="Path to uploaded file for processing")

class KnowledgeGraphManager:
    """Handles graph persistence and advanced querying"""
    def __init__(self):
        self.graph = NetworkxEntityGraph()
        self.storage_dir = "knowledge_graphs"
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_graph(self, graph_name: str):
        path = os.path.join(self.storage_dir, f"{graph_name}.graphml")
        nx.write_graphml(self.graph._graph, path)
        return path

    def load_graph(self, graph_name: str):
        path = os.path.join(self.storage_dir, f"{graph_name}.graphml")
        self.graph._graph = nx.read_graphml(path)
        return self.graph

class EnhancedKnowledgeGraphTool(BaseTool):
    name: str = "build_knowledge_graph"
    description: str = "Processes uploaded files to create and persist knowledge graphs"
    args_schema: Type[BaseModel] = RAGInput
    graph_manager = KnowledgeGraphManager()

    def _process_document(self, file_path: str) -> List[Document]:
        """Handle multiple file formats"""
        loader = None
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(('.txt', '.md')):
            loader = TextLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
            
        return loader.load()

    def _extract_entities(self, text: str) -> Doc:
        """Enhanced entity recognition with spaCy"""
        return nlp(text)

    def _run(self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Process file and construct knowledge graph with relationships"""
        try:
            # Load and split document
            documents = self._process_document(file_path)
            text_splitter = RecursiveCharacterTextSplitter()
            chunks = text_splitter.split_documents(documents)
            
            # Process chunks
            for chunk in chunks:
                doc = self._extract_entities(chunk.page_content)
                
                # Extract entities and relationships
                entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
                relations = self._extract_relationships(doc)
                
                # Add to graph
                for entity in entities:
                    self.graph_manager.graph.add_node(entity)
                for rel in relations:
                    self.graph_manager.graph.add_triplet(rel[0], rel[1], rel[2])
            
            # Create unique graph name
            graph_name = f"graph_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.graph_manager.save_graph(graph_name)
            
            # Create vector store
            self._create_vector_store(chunks, graph_name)
            
            return f"Knowledge graph '{graph_name}' created with {len(entities)} entities and {len(relations)} relations"
        except Exception as e:
            return f"Processing failed: {str(e)}"

    def _extract_relationships(self, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract relationships using dependency parsing"""
        relations = []
        for token in doc:
            if token.dep_ in ("dobj", "pobj"):
                subject = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                if subject:
                    relations.append((
                        subject[0].text,
                        token.dep_,
                        token.text
                    ))
        return relations

    def _create_vector_store(self, chunks: List[Document], graph_name: str):
        """Persist vector store with graph reference"""
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        vectorstore.save_local(f"{self.graph_manager.storage_dir}/{graph_name}_faiss")

class EnhancedRAGQueryTool(BaseTool):
    name: str = "rag_query"
    description: str = "Hybrid search combining vector and graph retrieval"
    args_schema: Type[BaseModel] = RAGInput
    graph_manager = KnowledgeGraphManager()

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute hybrid search with graph context"""
        try:
            # Load relevant graph
            graph_name = self._find_relevant_graph(query)
            self.graph_manager.load_graph(graph_name)
            
            # Vector search
            vector_results = self._vector_search(query, graph_name)
            
            # Graph search
            graph_results = self._graph_search(query)
            
            # Combine results
            combined = self._rank_results(vector_results, graph_results)
            
            return json.dumps(combined, indent=2)
        except Exception as e:
            return f"Query failed: {str(e)}"

    def _find_relevant_graph(self, query: str) -> str:
        """Find most relevant graph using vector similarity"""
        # Implementation for finding best matching graph
        return "latest_graph"  # Simplified for example

    def _vector_search(self, query: str, graph_name: str) -> List[dict]:
        """Search in FAISS vector store"""
        vectorstore = FAISS.load_local(
            f"{self.graph_manager.storage_dir}/{graph_name}_faiss",
            OpenAIEmbeddings()
        )
        docs = vectorstore.similarity_search(query, k=3)
        return [{"source": "vector", "content": doc.page_content} for doc in docs]

    def _graph_search(self, query: str) -> List[dict]:
        """Search in knowledge graph"""
        doc = nlp(query)
        entities = [ent.text for ent in doc.ents]
        results = []
        
        for entity in entities:
            if entity in self.graph_manager.graph._graph.nodes:
                neighbors = list(self.graph_manager.graph._graph.neighbors(entity))
                for neighbor in neighbors:
                    edge_data = self.graph_manager.graph._graph.get_edge_data(entity, neighbor)
                    results.append({
                        "source": "graph",
                        "content": f"{entity} - {edge_data['label']} - {neighbor}"
                    })
        return results

    def _rank_results(self, vector: List[dict], graph: List[dict]) -> List[dict]:
        """Combine and rank results using simple scoring"""
        combined = vector + graph
        # Add scoring logic based on content length and source
        return sorted(combined, key=lambda x: len(x['content']), reverse=True)[:5]

class RAGAgent:
    def __init__(self, llm, name: str, prompt: str):
        self._name = name
        self.rag_agent = create_react_agent(
            llm, 
            tools=[EnhancedKnowledgeGraphTool(), EnhancedRAGQueryTool()],
            prompt=prompt
        )

    def node(self, state: State):
        result = self.rag_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content, 
                        name=self._name
                    )
                ]
            },
            goto="shiami",
        )
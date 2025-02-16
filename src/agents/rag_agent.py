from typing import Type, Optional, List, Tuple, ClassVar
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import networkx as nx
import spacy
from spacy.tokens import Doc
import os
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, MPNetModel

# Core components
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import state
from langchain.agents import create_react_agent
from langchain.schema import HumanMessage
from langgraph.types import Command

# Initialize spaCy with English medium model
nlp = spacy.load("en_core_web_md")

# Use MPNetModel for embeddings rather than a masked LM
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = MPNetModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

class MPNetEmbeddings:
    """
    A helper class to generate text embeddings using MPNetModel.
    It provides methods to embed single queries or a list of documents.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        pooled_embedding = embeddings.mean(dim=1)  # (batch_size, hidden_dim)
        return pooled_embedding.squeeze().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
    
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

class RAGInput(BaseModel):
    query: Optional[str] = Field(None, description="User query for RAG retrieval")
    file_path: Optional[str] = Field(None, description="Path to uploaded file for processing")

class KnowledgeGraphManager:
    """Handles graph persistence and advanced querying."""
    def __init__(self):
        self.graph = NetworkxEntityGraph()
        self.storage_dir = "knowledge_graphs"
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def add_triple(self, subject: str, predicate: str, object_: str):
        """Add a triple to the graph with proper error handling."""
        try:
            self.graph.add_node(subject)
            self.graph.add_node(object_)
            self.graph._graph.add_edge(subject, object_, relation=predicate)
        except Exception as e:
            print(f"Error adding triple: {e}")

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
    graph_manager: ClassVar[KnowledgeGraphManager] = KnowledgeGraphManager()

    def _process_document(self, file_path: str) -> List[Document]:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(('.txt', '.md')):
            loader = TextLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        return loader.load()

    def _run(self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            documents = self._process_document(file_path)
            text_splitter = RecursiveCharacterTextSplitter()
            chunks = text_splitter.split_documents(documents)
            
            total_entities = set()
            total_relations = []
            for chunk in chunks:
                doc = nlp(chunk.page_content)
                entities = {ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']}
                relations = self._extract_relationships(doc)
                total_entities.update(entities)
                total_relations.extend(relations)
                for entity in entities:
                    self.graph_manager.graph.add_node(entity)
                for subj, pred, obj in relations:
                    self.graph_manager.add_triple(subj, pred, obj)
            
            graph_name = f"graph_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.graph_manager.save_graph(graph_name)
            self._create_vector_store(chunks, graph_name)
            
            return f"Knowledge graph '{graph_name}' created with {len(total_entities)} entities and {len(total_relations)} relations"
        except Exception as e:
            return f"Processing failed: {str(e)}"

    def _extract_relationships(self, doc: Doc) -> List[Tuple[str, str, str]]:
        relations = []
        for token in doc:
            if token.dep_ in ("dobj", "pobj"):
                subjects = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                if subjects:
                    subject = subjects[0].text
                    predicate = token.head.lemma_
                    object_ = token.text
                    relations.append((subject, predicate, object_))
        return relations

    def _create_vector_store(self, chunks: List[Document], graph_name: str):
        mpnet_embeddings = MPNetEmbeddings(model, tokenizer)
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=mpnet_embeddings
        )
        vectorstore.save_local(f"{self.graph_manager.storage_dir}/{graph_name}_faiss")

class EnhancedRAGQueryTool(BaseTool):
    name: str = "rag_query"
    description: str = "Hybrid search combining vector and graph retrieval"
    args_schema: Type[BaseModel] = RAGInput
    graph_manager: ClassVar[KnowledgeGraphManager] = KnowledgeGraphManager()

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            graph_name = self._find_relevant_graph(query)
            self.graph_manager.load_graph(graph_name)
            vector_results = self._vector_search(query, graph_name)
            graph_results = self._graph_search(query)
            combined = self._rank_results(vector_results, graph_results)
            return json.dumps(combined, indent=2)
        except Exception as e:
            return f"Query failed: {str(e)}"

    def _find_relevant_graph(self, query: str) -> str:
        graph_files = [f for f in os.listdir(self.graph_manager.storage_dir) if f.endswith('.graphml')]
        if not graph_files:
            raise Exception("No knowledge graphs found")
        return graph_files[-1].replace('.graphml', '')

    def _vector_search(self, query: str, graph_name: str) -> List[dict]:
        mpnet_embeddings = MPNetEmbeddings(model, tokenizer)
        vectorstore = FAISS.load_local(
            f"{self.graph_manager.storage_dir}/{graph_name}_faiss",
            mpnet_embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(query, k=3)
        return [{"source": "vector", "content": doc.page_content} for doc in docs]

    def _graph_search(self, query: str) -> List[dict]:
        doc = nlp(query)
        entities = [ent.text for ent in doc.ents]
        results = []
        for entity in entities:
            if entity in self.graph_manager.graph._graph.nodes:
                for neighbor in self.graph_manager.graph._graph.neighbors(entity):
                    edge_data = self.graph_manager.graph._graph.get_edge_data(entity, neighbor)
                    results.append({
                        "source": "graph",
                        "content": f"{entity} - {edge_data.get('relation', 'related_to')} - {neighbor}"
                    })
        return results

    def _rank_results(self, vector: List[dict], graph: List[dict]) -> List[dict]:
        combined = vector + graph
        return sorted(combined, key=lambda x: len(x["content"]), reverse=True)[:5]

class RAGAgent:
    def __init__(self, llm, name: str, prompt):
        self._name = name
        self.tools = [EnhancedKnowledgeGraphTool(), EnhancedRAGQueryTool()]
        prompt_template = PromptTemplate.from_template(
            """You have access to the following tools:
{tools}

Use the following format:

Question: {input}
Thought: {agent_scratchpad}
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (repeat Thought/Action/Action Input/Observation N times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question

Begin!

Question: {input}"""
        )
        self.rag_agent = create_react_agent(
            llm=llm,
            tools=self.tools,
            prompt=prompt_template
        )

    def invoke(self, user_input: str):
        # Provide empty placeholders for scratchpad and intermediate_steps.
        result = self.rag_agent.invoke({
            "input": user_input,
            "agent_scratchpad": "",
            "intermediate_steps": []
        })
        messages = result.messages
        if not messages:
            return "No response"
        final_msg = messages[-1]
        # First, if the final message has an observation (common for AgentAction), use that.
        if hasattr(final_msg, "observation") and final_msg.observation:
            return final_msg.observation
        # Otherwise, if it has a content attribute, use that.
        elif hasattr(final_msg, "content") and final_msg.content:
            return final_msg.content
        # If neither is available, try to use a log attribute.
        elif hasattr(final_msg, "log") and final_msg.log:
            return final_msg.log
        else:
            # Fallback: convert the final message to string.
            return str(final_msg)

    
    def node(self, state):
        result = self.invoke(state["messages"][-1].content)
        return Command(
            update={"messages": [{"content": result, "name": self._name}]},
            goto="next_node"
        )
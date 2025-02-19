import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import List, Tuple, Dict, Any
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
import heapq
import argparse
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from spacy.cli import download
from helper_functions import *

load_dotenv()

# Set the OpenAI API key environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def rate_limit(min_delay: float = 5.0):
    """Decorator to enforce minimum delay between function calls"""
    last_call_time = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            last_time = last_call_time.get(func.__name__, 0)
            time_to_wait = max(0, min_delay - (current_time - last_time))
            
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            last_call_time[func.__name__] = time.time()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class GeminiCallbackHandler:
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        pass
        
    def update_counts(self, prompt_tokens, completion_tokens):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens



class DocumentProcessor:
    def __init__(self):
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.process_texts = []
        self.original_texts =[]
    
    def preprocess_text(self, text):
        
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
        return tokens
    
    def process_documents(self, documents):
        
        splits = self.text_splitter.split_documents(documents)
        self.original_texts = [split.page_content for split in splits]
        
        vector_store = FAISS.from_documents(splits, self.embeddings)
        
        processed_texts = [self.preprocess_text(split.page_content) for split in splits]
        bm25 = BM25Okapi(processed_texts)

        self.process_texts = processed_texts
        tfidf_matrix = self.tfidf.fit_transform([split.page_content for split in splits])
        
        return splits, vector_store, bm25, tfidf_matrix, self.original_texts
    
    @rate_limit(min_delay=5.0)
    def create_embeddings_batch(self, texts, batch_size=32):
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
       
        return cosine_similarity(embeddings)


class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

    
    def _extract_concepts_statistical(self, content, tfidf_matrix, tfidf_vectorizer, top_n=5):
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]
        
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        important_terms = [feature_names[i] for i in tfidf_scores.argsort()[-top_n:]]
        
        all_concepts = list(set(named_entities + important_terms))
        return all_concepts[:top_n]  # Return top N concepts

    def build_graph(self, splits, tfidf_matrix, tfidf_vectorizer, embedding_model):
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        
        for i, split in enumerate(splits):
            concepts = self._extract_concepts_statistical(
                split.page_content, 
                tfidf_matrix[i], 
                tfidf_vectorizer
            )
            self.graph.nodes[i]['concepts'] = concepts
            
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        
    @rate_limit(min_delay=5.0)
    def _extract_concepts_and_entities(self, content, llm):
        if content in self.concept_cache:
            return self.concept_cache[content]

        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total=len(splits),
                               desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(
                        self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])

class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer to the query")
    answer: str = Field(description="The current answer based on the context, if any")


class QueryEngine:
    def __init__(self, vector_store, bm25, knowledge_graph, llm, document_processor,corpus):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 30000
        self.document_processor = document_processor
        self.corpus = corpus
        self.answer_check_chain = self._create_answer_check_chain()

    def _find_matching_node(self, content):

        try:
            for node in self.knowledge_graph.graph.nodes():
                node_content = self.knowledge_graph.graph.nodes[node].get('content', '')
                if content.strip() == node_content.strip():
                    return node
                
            max_similarity = 0
            best_match = None
            
            for node in self.knowledge_graph.graph.nodes():
                node_content = self.knowledge_graph.graph.nodes[node].get('content', '')
                # Calculate similarity using a simple ratio
                similarity = len(set(content.split()) & set(node_content.split())) / len(set(content.split()))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = node
                    
            return best_match if max_similarity > 0.5 else None
            
        except Exception as e:
            print(f"Error in finding matching node: {e}")
            return None
            
    def _hybrid_search(self, query, k=5):
        """Combine semantic and keyword search results"""
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")

            semantic_results = self.vector_store.similarity_search(query, k=k)  # Removed _with_score
            
            processed_query = self.document_processor.preprocess_text(query)
            bm25_scores = self.bm25.get_scores(processed_query)
            top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
            
            combined_results = []
            seen_content = set()
            
            for doc in semantic_results:  # Removed score unpacking
                if doc.page_content not in seen_content:
                    combined_results.append(doc)
                    seen_content.add(doc.page_content)
            
            for idx in top_bm25_indices:
                content = self.corpus[idx]
                if content not in seen_content:
                    combined_results.append(Document(
                        page_content=content,
                        metadata={"source": "bm25", "score": bm25_scores[idx]}
                    ))
                    seen_content.add(content)
            return combined_results[:k]
        
        except Exception as e:
            print(f"Error during hybrid search: {e}")
            return self.vector_store.similarity_search(query, k=k)
        

    def _create_answer_check_chain(self):
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)
    
    @rate_limit(min_delay=5.0)
    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
       
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
       
        # Initialize variables
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}  # Stores the best known "distance" (inverse of connection strength) to each node

        print("\nTraversing the knowledge graph:")

        # Initialize priority queue with closest nodes from relevant docs
        for doc in relevant_docs:
            # Find the most similar node in the knowledge graph for each relevant document
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            # Get the corresponding node in our knowledge graph
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            # Initialize priority (inverse of similarity score for min-heap behavior)
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0
        while priority_queue:
            # Get the node with the highest priority (lowest distance value)
            current_priority, current_node = heapq.heappop(priority_queue)

            # Skip if we've already found a better path to this node
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # Add node content to our accumulated context
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # Log the current step for debugging and visualization
                print(f"\nStep {step} - Node {current_node}:")
                print(f"Content: {node_content[:100]}...")
                print(f"Concepts: {', '.join(node_concepts)}")
                print("-" * 50)

                # Check if we have a complete answer with the current context
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                # Process the concepts of the current node
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # Explore neighbors
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # Calculate new distance (priority) to the neighbor
                        # Note: We use 1 / edge_weight because higher weights mean stronger connections
                        distance = current_priority + (1 / edge_weight)

                        # If we've found a stronger connection to the neighbor, update its distance
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # Process the neighbor node if it's not already in our traversal path
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                # Log the neighbor node information
                                print(f"\nStep {step} - Node {neighbor} (neighbor of {current_node}):")
                                print(f"Content: {neighbor_content[:100]}...")
                                print(f"Concepts: {', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # Check if we have a complete answer after adding the neighbor's content
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                # Process the neighbor's concepts
                                neighbor_concepts_set = set(
                                    self.knowledge_graph._lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # If we found a final answer, break out of the main loop
                if final_answer:
                    break

        # If we haven't found a complete answer, generate one using the LLM
        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer
    
    @rate_limit(min_delay=2.0)  # Increased delay for LLM calls

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """Modified query method with improved error handling"""
        print(f"\nProcessing query: {query}")
        
        try:
            # Use hybrid search instead of pure semantic search
            relevant_docs = self._hybrid_search(query)
            
            if not relevant_docs:
                return "No relevant documents found.", [], {}
            
            # Use graph traversal to expand context
            expanded_context, traversal_path, filtered_content = self._expand_context_statistical(
                query, relevant_docs
            )
            
            if not expanded_context:
                return "Could not expand context.", [], {}
            
            # Single LLM call for final answer generation
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, provide a comprehensive answer to the query. If the information is not available in the context, say so. Context: {context}\n\nQuery: {query}\n\nAnswer:"
            )
            
            response_chain = response_prompt | self.llm
            final_answer = response_chain.invoke({
                "query": query,
                "context": expanded_context
            })
            
            return final_answer, traversal_path, filtered_content
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return f"An error occurred while processing the query: {str(e)}", [], {}
        
    def _retrieve_relevant_documents(self, query: str):
       
        print("\nRetrieving relevant documents...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        return compression_retriever.invoke(query)
    
    def _expand_context_statistical(self, query, relevant_docs, max_hops=2):
        """Expand context using statistical relationships with improved error handling"""
        expanded_context = []
        traversal_path = []
        filtered_content = {}
        seen_nodes = set()

        if not relevant_docs:
            return "", [], {}

        try:
            # Process each relevant document
            for doc in relevant_docs:
                # Find matching node in knowledge graph
                start_node = self._find_matching_node(doc.page_content)
                if start_node is None:
                    continue

                # Perform BFS traversal
                queue = [(start_node, 0)]  # (node, hop_count)
                while queue:
                    current_node, hops = queue.pop(0)
                    
                    if current_node in seen_nodes or hops > max_hops:
                        continue
                    
                    seen_nodes.add(current_node)
                    traversal_path.append(current_node)
                    
                    # Add node content to expanded context
                    node_content = self.knowledge_graph.graph.nodes[current_node].get('content', '')
                    if node_content:
                        expanded_context.append(node_content)
                        filtered_content[current_node] = node_content

                    # Add neighbors to queue
                    if hops < max_hops:
                        neighbors = list(self.knowledge_graph.graph.neighbors(current_node))
                        queue.extend((neighbor, hops + 1) for neighbor in neighbors)

            # Join expanded context with newlines
            final_context = "\n".join(expanded_context) if expanded_context else ""
            
            return final_context, traversal_path, filtered_content

        except Exception as e:
            print(f"Error in context expansion: {e}")
            # Return original documents as fallback
            return "\n".join(doc.page_content for doc in relevant_docs), [], {}

# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define the Visualizer class
class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        
        traversal_graph = nx.DiGraph()

        # Add nodes and edges from the original graph
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate positions for all nodes
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # Draw regular edges with color based on weight
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)

        # Draw traversal path with curved arrows
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # Calculate control point for curve
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # Draw curved arrow
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',
                                            linewidth=2,
                                            zorder=4)
            ax.add_patch(arrow)

        # Prepare labels for the nodes
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # Draw labels
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        # Highlight start and end nodes
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[start_node],
                               node_color='lightgreen',
                               node_size=3000,
                               ax=ax)

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[end_node],
                               node_color='lightcoral',
                               node_size=3000,
                               ax=ax)

        ax.set_title("Graph Traversal Flow")
        ax.axis('off')

        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                   norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)

        # Add legend
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Regular Edge')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15,
                                 label='Start Node')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15,
                               label='End Node')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left',
                            bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        
        print("\nFiltered content of visited nodes in order of traversal:")
        for i, node in enumerate(traversal_path):
            print(f"\nStep {i + 1} - Node {node}:")
            print(
                f"Filtered Content: {filtered_content.get(node, 'No filtered content available')[:200]}...")  # Print first 200 characters
            print("-" * 50)


# Define the graph RAG class
class GraphRAG:
    def __init__(self, documents):
       
        self.llm =ChatGoogleGenerativeAI(temperature=0.5, model="gemini-2.0-flash")
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None
        self.visualizer = Visualizer()
        self.process_documents(documents)

    def process_documents(self, documents):
        # Process documents using both vector store and BM25
        splits, vector_store, bm25, tfidf_matrix , corpus = self.document_processor.process_documents(documents)
        
        # Build knowledge graph using statistical methods
        self.knowledge_graph.build_graph(
            splits, 
            tfidf_matrix,
            self.document_processor.tfidf,
            self.embedding_model
        )
        
        # Initialize query engine with both search methods
        self.query_engine = QueryEngine(
            vector_store,
            bm25,
            self.knowledge_graph,
            self.llm,
            document_processor = self.document_processor,
            corpus = corpus
        )

    def query(self, query: str):
        
        try:
            # Get response from query engine
            print(f"\nProcessing query: {query}")
            response, traversal_path, filtered_content = self.query_engine.query(query)
            
            # Print the initial response
            print("\nResponse received:")
            print("-" * 50)
            print(response)
            print("-" * 50)

            # Visualize the traversal if available
            if traversal_path:
                print("\nVisualizing traversal path...")
                self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
                
                # Print detailed content
                print("\nDetailed content traversal:")
                self.visualizer.print_filtered_content(traversal_path, filtered_content)
            else:
                print("\nNo traversal path to visualize.")

            return response, traversal_path, filtered_content
            
        except Exception as e:
            print(f"\nError in query processing: {e}")
            return f"An error occurred: {str(e)}", [], {}


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="GraphRAG system")
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help='Path to the PDF file.')
    parser.add_argument('--query', type=str, default='what is the main cause of climate change?',
                        help='Query to retrieve documents.')
    return parser.parse_args()

def run_query_with_visualization(graph_rag, query):
    
    print("\n" + "="*50)
    print(f"Processing query: {query}")
    print("="*50)
    
    try:
        # Get response and visualization data
        response, traversal_path, filtered_content = graph_rag.query_engine.query(query)
        
        # Print the response
        print("\nResponse:")
        print("-"*50)
        print(response)
        print("-"*50)
        
        # Visualize if there's a traversal path
        if traversal_path:
            print("\nGenerating visualization...")
            graph_rag.visualizer.visualize_traversal(
                graph_rag.knowledge_graph.graph, 
                traversal_path
            )
            
            # Print filtered content
            print("\nDocument traversal details:")
            graph_rag.visualizer.print_filtered_content(
                traversal_path, 
                filtered_content
            )
        else:
            print("\nNo traversal path to visualize.")
            
        return response
        
    except Exception as e:
        print(f"\nError during query processing: {e}")
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    try:
        # Load the documents
        print("\nLoading documents...")
        loader = PyPDFLoader(args.path)
        documents = loader.load()
        documents = documents[:10]  # Limit to first 10 documents for testing

        # Create and initialize graph RAG
        print("\nInitializing GraphRAG...")
        graph_rag = GraphRAG(documents)
        
        # Process documents
        print("\nProcessing documents...")
        graph_rag.process_documents(documents)

        # Run query and get results
        result = run_query_with_visualization(graph_rag, args.query)
        
        # Final summary
        print("\n" + "="*50)
        print("Query Processing Complete")
        print("="*50)
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        print("Please check your input files and arguments.")

# TO run poetry run python rag_agent.py --path "path to pdf" --query "query to be asked"
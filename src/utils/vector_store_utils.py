import os
import faiss
import uuid
import logging
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger("utils/vector_store_utils")

class VectorStoreUtils:
    def __init__(self, tools: list):
        self.tools = tools
        self.vector_store = self.create_vector_store()

    def create_vector_store(self) -> FAISS:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            
            # Initialize FAISS index
            embedding_dim = len(embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(embedding_dim)

            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            # Convert tools into Document format
            tool_documents = []
            for tool in self.tools:
                try:
                    doc = Document(
                        page_content=tool.description,
                        id=str(uuid.uuid4()),  # Assign unique IDs
                        metadata={"tool_name": tool.name},
                    )
                    tool_documents.append(doc)
                except Exception as e:
                    logger.error(f"Error creating document for tool {tool.name}: {str(e)}")

            # Add tool documents to vector store
            if tool_documents:
                vector_store.add_documents(tool_documents)

            return vector_store
        except Exception as e:
            logger.error(f"Error in creating vector store: {str(e)}")
            raise

    def route_tools(self, query: str, tool_registry: dict) -> list[str]:
        try:
            tool_documents = self.vector_store.similarity_search(query, k=3)
            selected_tool_ids = [
                doc.metadata["tool_name"] for doc in tool_documents if doc.metadata["tool_name"] in tool_registry
            ]

            if not selected_tool_ids:
                selected_tool_ids = list(tool_registry.keys())

            return selected_tool_ids
        except Exception as e:
            logger.error(f"Error in route_tools: {str(e)}")
            return list(tool_registry.keys())

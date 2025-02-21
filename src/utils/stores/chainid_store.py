import faiss
import uuid
import json
import logging
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger("utils/stores/chainid_store")


class ChainIDStore:
    def __init__(self):
        self.chain_id_map = {
            "Arbitrum": {"chainId": 42161, "nativeToken": "ETH"},
            "Avalanche": {"chainId": 43114, "nativeToken": "AVAX"},
            "BNB Chain": {"chainId": 56, "nativeToken": "BNB"},
            "Ethereum": {"chainId": 1, "nativeToken": "ETH"},
            "Polygon": {"chainId": 137, "nativeToken": "MATIC"},
            "Solana": {"chainId": 7565164, "nativeToken": "SOL"},
            "Linea": {"chainId": 59144, "nativeToken": "ETH"},
            "Base": {"chainId": 8453, "nativeToken": "ETH"},
            "Optimism": {"chainId": 10, "nativeToken": "ETH"},
            "Neon": {"chainId": 100000001, "nativeToken": "NEON"},
            "Gnosis": {"chainId": 100000002, "nativeToken": "xDAI"},
            "Metis": {"chainId": 100000004, "nativeToken": "METIS"},
            "Bitrock": {"chainId": 100000005, "nativeToken": "BROCK"},
            "Sonic": {"chainId": 100000014, "nativeToken": "S"},
            "CrossFi": {"chainId": 100000006, "nativeToken": "XFI"},
            "Cronos zkEVM": {"chainId": 100000010, "nativeToken": "zkCRO"},
            "Abstract": {"chainId": 100000017, "nativeToken": "ETH"},
            "Berachain": {"chainId": 100000020, "nativeToken": "BERA"},
            "Story": {"chainId": 100000013, "nativeToken": "IP"},
            "HyperEVM": {"chainId": 100000022, "nativeToken": "WHYPE"},
        }

        self.vector_store = self.create_vector_store()

    def create_vector_store(self) -> FAISS:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004")

            # Initialize FAISS index
            embedding_dim = len(embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(embedding_dim)

            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            # Convert chain names into Document format
            chain_documents = []
            for chain_name, data in self.chain_id_map.items():
                try:
                    doc = Document(
                        page_content=json.dumps(
                            {"chainName": chain_name, "chainId": data["chainId"], "nativeToken": data["nativeToken"]}),
                        id=str(uuid.uuid4()),  # Assign unique IDs
                        metadata={"chainId": data["chainId"],
                                  "nativeToken": data["nativeToken"]},
                    )
                    chain_documents.append(doc)
                except Exception as e:
                    logger.error(
                        f"Error creating document for chain {chain_name}: {str(e)}")

            # Add chain documents to vector store
            if chain_documents:
                vector_store.add_documents(chain_documents)

            return vector_store
        except Exception as e:
            logger.error(f"Error in creating vector store: {str(e)}")
            raise

    def get_chain_id(self, chain_name: str) -> int:
        try:
            chain_documents = self.vector_store.similarity_search(
                chain_name, k=1)
            if chain_documents:
                return chain_documents[0].metadata["chainId"]
            else:
                logger.warning(f"Chain name '{chain_name}' not found")
                return None
        except Exception as e:
            logger.error(f"Error in get_chain_id: {str(e)}")
            raise

    def is_native_token(self, chain_name: str, token: str) -> bool:
        try:
            chain_documents = self.vector_store.similarity_search(
                chain_name, k=1)
            if chain_documents:
                return chain_documents[0].metadata["nativeToken"].upper() == token.upper()
            else:
                logger.warning(f"Chain name '{chain_name}' not found")
                return False
        except Exception as e:
            logger.error(f"Error in is_native_token: {str(e)}")
            raise

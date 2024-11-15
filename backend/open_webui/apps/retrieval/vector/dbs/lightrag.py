import asyncio
import os
import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Type, Optional, List, Dict, Union
from logging import getLogger, StreamHandler, Formatter, DEBUG

from lightrag import LightRAG  # Ensure LightRAG is imported correctly
from lightrag.llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)

from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    local_query,
    global_query,
    hybrid_query,
    naive_query,
)
from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from lightrag.storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

# Future KG integrations
# from lightrag.kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )

# Wrapper for embedding function
def embedding_func_wrapper(texts: List[str], *args, **kwargs) -> List[np.ndarray]:
    """
    Wrapper for the embedding function.

    Parameters:
    - texts: List of text strings to generate embeddings for.
    - args, kwargs: Additional arguments for customization.

    Returns:
    - List of embedding vectors as numpy arrays.
    """
    logger.debug("Using OpenAI embeddings via embedding_func_wrapper...")
    return openai_embedding(texts, *args, **kwargs)

# Wrapper for LLM function
def llm_func_wrapper(prompt: str, *args, **kwargs) -> str:
    """
    Wrapper for the LLM function.

    Parameters:
    - prompt: Input prompt for the language model.
    - args, kwargs: Additional arguments for customization.

    Returns:
    - Response from the LLM as a string.
    """
    logger.debug("Using GPT-4 mini via llm_func_wrapper...")
    return gpt_4o_mini_complete(prompt, *args, **kwargs)

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Retrieve the current event loop or create a new one if none exists.

    Returns:
        asyncio.AbstractEventLoop: The event loop.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

@dataclass
class QueryResult:
    """
    Dataclass to store query results.

    Attributes:
        ids (List[List[str]]): Nested list of document IDs.
        embeddings (Optional[List[np.ndarray]]): List of document embeddings (optional).
    """
    ids: List[List[str]]  # Nested list to match usage result.ids[0]
    embeddings: Optional[List[np.ndarray]] = None  # Added embeddings field

@dataclass
class LightRAGClient:
    """
    Client class to interact with LightRAG instances for various operations like insert, search, and query.

    Attributes:
        working_dir (str): Directory for storing LightRAG cache.
        kv_storage (str): Key-Value storage backend.
        vector_storage (str): Vector storage backend.
        graph_storage (str): Graph storage backend.
        log_level (str): Logging level.
        chunk_token_size (int): Token size for text chunking.
        chunk_overlap_token_size (int): Overlap token size for text chunking.
        tiktoken_model_name (str): Model name for tokenization.
        entity_extract_max_gleaning (int): Max gleaning for entity extraction.
        entity_summary_to_max_tokens (int): Max tokens for entity summary.
        node_embedding_algorithm (str): Algorithm for node embedding.
        node2vec_params (Dict): Parameters for node2vec algorithm.
        embedding_func (EmbeddingFunc): Embedding function wrapper.
        llm_model_func (callable): LLM function wrapper.
        embedding_batch_num (int): Batch number for embeddings.
        embedding_func_max_async (int): Max async calls for embedding function.
        llm_model_max_token_size (int): Max token size for LLM model.
        llm_model_max_async (int): Max async calls for LLM model.
        llm_model_kwargs (Dict): Additional kwargs for LLM model.
        vector_db_storage_cls_kwargs (Dict): Additional kwargs for vector DB storage.
        enable_llm_cache (bool): Flag to enable LLM cache.
        addon_params (Dict): Additional parameters for extensions.
        convert_response_to_json_func (callable): Function to convert responses to JSON.
        collections (Dict[str, 'LightRAG']): Dictionary to store LightRAG instances per collection.
    """
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # Text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: Dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # Embedding and LLM functions
    embedding_func: EmbeddingFunc = field(default_factory=lambda: EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embedding_func_wrapper
    ))
    llm_model_func: callable = field(default_factory=lambda: llm_func_wrapper)

    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: Dict = field(default_factory=dict)

    # Storage
    vector_db_storage_cls_kwargs: Dict = field(default_factory=dict)
    enable_llm_cache: bool = True

    # Extension
    addon_params: Dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # Storage Instances (Initialized in __post_init__)
    collections: Dict[str, 'LightRAG'] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """
        Post-initialization to set up logging and initialize the collections dictionary.
        """
        # Configure logging to output to stdout
        console_handler = StreamHandler()
        console_handler.setLevel(self.log_level)
        formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger_obj = getLogger(__name__)
        logger_obj.setLevel(self.log_level)
        # Remove existing handlers to prevent duplicate logs
        if logger_obj.hasHandlers():
            logger_obj.handlers.clear()
        logger_obj.addHandler(console_handler)

        logger_obj.info("Logger initialized for console output.")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger_obj.debug(f"LightRAGClient initialized with parameters:\n  {_print_config}\n")

        # Initialize collections dict
        self.collections: Dict[str, 'LightRAG'] = {}

    def _get_storage_class(self) -> Dict[str, Type]:
        """
        Retrieve a dictionary mapping storage class names to their corresponding classes.

        Returns:
            Dict[str, Type]: Mapping of storage class names to classes.
        """
        return {
            # KV storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            # Vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            # Graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage  # Uncomment if needed
        }

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        return collection_name in self.collections

    def get_lightRAG_instance(self, collection_name: str) -> 'LightRAG':
        """
        Retrieve or initialize a LightRAG instance for the specified collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            LightRAG: The LightRAG instance for the collection.
        """
        # Debug logs for investigation
        logger.debug(f"self.working_dir: {self.working_dir}, type: {type(self.working_dir)}")
        logger.debug(f"collection_name: {collection_name}, type: {type(collection_name)}")

        # Ensure working_dir and collection_name are strings
        if not isinstance(self.working_dir, str):
            logger.error(f"working_dir is not a string: {self.working_dir} ({type(self.working_dir)})")
            raise TypeError(f"working_dir must be a string, got {type(self.working_dir)}")

        if not isinstance(collection_name, str):
            logger.error(f"collection_name is not a string: {collection_name} ({type(collection_name)})")
            raise TypeError(f"collection_name must be a string, got {type(collection_name)}")

        collection_working_dir = os.path.join(self.working_dir, collection_name)
        logger.debug(f"collection_working_dir: {collection_working_dir}")

        # Ensure the directory exists
        try:
            os.makedirs(collection_working_dir, exist_ok=True)
            logger.debug(f"Ensured existence of directory: {collection_working_dir}")
        except Exception as e:
            logger.error(f"Failed to create directory {collection_working_dir}: {e}")
            raise

        if collection_name not in self.collections:
            # Initialize a new LightRAG instance for the collection
            try:
                lightrag_instance = LightRAG(
                    working_dir=collection_working_dir,
                    kv_storage=self.kv_storage,
                    vector_storage=self.vector_storage,
                    graph_storage=self.graph_storage,
                    log_level=self.log_level,
                    chunk_token_size=self.chunk_token_size,
                    chunk_overlap_token_size=self.chunk_overlap_token_size,
                    tiktoken_model_name=self.tiktoken_model_name,
                    entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                    entity_summary_to_max_tokens=self.entity_summary_to_max_tokens,
                    node_embedding_algorithm=self.node_embedding_algorithm,
                    node2vec_params=self.node2vec_params,
                    embedding_func=self.embedding_func,
                    llm_model_func=self.llm_model_func,
                    embedding_batch_num=self.embedding_batch_num,
                    embedding_func_max_async=self.embedding_func_max_async,
                    llm_model_max_token_size=self.llm_model_max_token_size,
                    llm_model_max_async=self.llm_model_max_async,
                    llm_model_kwargs=self.llm_model_kwargs,
                    vector_db_storage_cls_kwargs=self.vector_db_storage_cls_kwargs,
                    enable_llm_cache=self.enable_llm_cache,
                    addon_params=self.addon_params,
                    convert_response_to_json_func=self.convert_response_to_json_func,
                )
                self.collections[collection_name] = lightrag_instance
                logger.info(f"Initialized LightRAG instance for collection '{collection_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize LightRAG instance for collection '{collection_name}': {e}")
                raise
        return self.collections[collection_name]

    def insert(self, items: Union[str, List[str]], *, collection_name: str = "default") -> bool:
        """
        Insert documents into the specified collection.

        Args:
            items (Union[str, List[str]]): The text(s) to insert.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            return lightrag_instance.insert(items)
        except Exception as e:
            logger.error(f"Failed to insert documents into collection '{collection_name}': {e}")
            return False

    async def ainsert(self, items: Union[str, List[str]], *, collection_name: str = "default") -> Optional[bool]:
        """
        Asynchronously insert documents into the specified collection.

        Args:
            items (Union[str, List[str]]): The text(s) to insert.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            Optional[bool]: True if insertion was successful, False otherwise.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            return await lightrag_instance.ainsert(items)
        except Exception as e:
            logger.error(f"Failed to asynchronously insert documents into collection '{collection_name}': {e}")
            return False

    def query(self, query: str = "", *, collection_name: str = "default", param: QueryParam = QueryParam(), filter: Optional[Dict] = None) -> Optional[Union[str, QueryResult]]:
        """
        Query the specified collection.

        Args:
            query (str, optional): The query string. Defaults to an empty string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            param (QueryParam, optional): Query parameters. Defaults to QueryParam().
            filter (Optional[Dict], optional): Filter criteria for the query. Defaults to None.

        Returns:
            Optional[Union[str, QueryResult]]: The query result as a string or QueryResult object.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, collection_name=collection_name, param=param, filter=filter))

    async def aquery(self, query: str = "", *, collection_name: str = "default", param: QueryParam = QueryParam(), filter: Optional[Dict] = None) -> Optional[Union[str, QueryResult]]:
        """
        Asynchronously query the specified collection.

        Args:
            query (str, optional): The query string. Defaults to an empty string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            param (QueryParam, optional): Query parameters. Defaults to QueryParam().
            filter (Optional[Dict], optional): Filter criteria for the query. Defaults to None.

        Returns:
            Optional[Union[str, QueryResult]]: The query result as a string or QueryResult object.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)

            if filter:
                # Simulate 'search_filtered' by integrating 'filter' into the query process
                logger.debug("Performing query with filter.")
                response = await lightrag_instance.aquery(query, param)
                # Assuming 'response' is a QueryResult or similar
                # Implement filtering based on 'filter' criteria

                # Example: Exclude documents whose IDs are in filter['excluded_ids']
                excluded_ids = filter.get('excluded_ids', [])
                if not isinstance(excluded_ids, list):
                    logger.warning("'excluded_ids' should be a list.")
                    excluded_ids = []

                # Filter out excluded IDs
                filtered_ids = [doc_id for doc_id in response.ids[0] if doc_id not in excluded_ids]
                logger.debug(f"Filtered out {len(response.ids[0]) - len(filtered_ids)} documents based on 'excluded_ids'.")
                return QueryResult(ids=[filtered_ids], embeddings=None)
            else:
                # Perform a standard generative query
                response = await lightrag_instance.aquery(query, param)
                logger.info(f"Query response: {response}")
                return QueryResult(ids=response.ids, embeddings=None)  # Adjust as needed
        except Exception as e:
            logger.error(f"Query failed for '{query}' with error: {e}")
            return None
        finally:
            await self._query_done(collection_name)

    def search(self, query: str, *, collection_name: str = "default", vectors: Optional[List[np.ndarray]] = None, limit: Optional[int] = None) -> Optional[QueryResult]:
        """
        Search the specified collection with optional vectors for similarity filtering and limit.

        Args:
            query (str): The search query string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            vectors (Optional[List[np.ndarray]], optional): Vectors to filter out similar documents. Defaults to None.
            limit (Optional[int], optional): Maximum number of results to return. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result containing document IDs.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.asearch(query=query, collection_name=collection_name, vectors=vectors, limit=limit))

    async def asearch(self, query: str, *, collection_name: str = "default", vectors: Optional[List[np.ndarray]] = None, limit: Optional[int] = None) -> Optional[QueryResult]:
        """
        Asynchronously search the specified collection with optional vectors for similarity filtering and limit.

        Args:
            query (str): The search query string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            vectors (Optional[List[np.ndarray]], optional): Vectors to filter out similar documents. Defaults to None.
            limit (Optional[int], optional): Maximum number of results to return. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result containing document IDs.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)

            # Perform the search using aquery
            response = await lightrag_instance.aquery(query, QueryParam())

            # Check if 'vectors' are provided for similarity filtering
            if vectors:
                logger.debug("Performing similarity filtering based on provided vectors.")
                # Fetch embeddings for the retrieved document IDs
                document_embeddings = self._get_document_embeddings(response.ids[0], collection_name)

                # Compute cosine similarity between provided vectors and document embeddings
                # Exclude documents that are similar to any of the provided vectors above a threshold
                threshold = 0.8  # Example threshold; adjust as needed
                filtered_ids = []

                for doc_id, doc_embedding in zip(response.ids[0], document_embeddings):
                    if not any(self._cosine_similarity(vec, doc_embedding) > threshold for vec in vectors):
                        filtered_ids.append(doc_id)

                logger.debug(f"Filtered out {len(response.ids[0]) - len(filtered_ids)} documents based on similarity.")

                # Apply the limit if specified
                if limit is not None:
                    filtered_ids = filtered_ids[:limit]
                    logger.debug(f"Applied limit: returning {len(filtered_ids)} documents.")

                return QueryResult(ids=[filtered_ids], embeddings=None)
            else:
                # If no vectors provided, return the search results as is, applying the limit if specified
                logger.debug("No vectors provided for similarity filtering. Returning all search results.")
                final_ids = response.ids[0]
                if limit is not None:
                    final_ids = final_ids[:limit]
                    logger.debug(f"Applied limit: returning {len(final_ids)} documents.")
                return QueryResult(ids=[final_ids], embeddings=None)
        except Exception as e:
            logger.error(f"Search failed for '{query}' with error: {e}")
            return None
        finally:
            await self._query_done(collection_name)

    def _get_document_embeddings(self, doc_ids: List[str], collection_name: str) -> List[np.ndarray]:
        """
        Retrieve embeddings for the given document IDs.

        Args:
            doc_ids (List[str]): List of document IDs.
            collection_name (str): Name of the collection.

        Returns:
            List[np.ndarray]: List of document embeddings.
        """
        embeddings = []
        lightrag_instance = self.get_lightRAG_instance(collection_name)
        for doc_id in doc_ids:
            embedding = lightrag_instance.get_embedding(doc_id)  # Ensure this method exists
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Embedding not found for document ID: {doc_id}. Using zero vector.")
                embeddings.append(np.zeros(self.embedding_func.embedding_dim))
        return embeddings

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity.
        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            logger.warning("One or both vectors are not numpy arrays.")
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def delete_by_entity(self, entity_name: str, *, collection_name: str = "default") -> bool:
        """
        Delete an entity and its relationships from the specified collection.

        Args:
            entity_name (str): The name of the entity to delete.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name, collection_name=collection_name))

    async def adelete_by_entity(self, entity_name: str, *, collection_name: str = "default") -> bool:
        """
        Asynchronously delete an entity and its relationships from the specified collection.

        Args:
            entity_name (str): The name of the entity to delete.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            return await lightrag_instance.adelete_by_entity(entity_name)
        except Exception as e:
            logger.error(f"Failed to asynchronously delete entity '{entity_name}' from collection '{collection_name}': {e}")
            return False

    async def _insert_done(self, collection_name: str):
        """
        Handle post-insertion operations.

        Args:
            collection_name (str): The name of the collection.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance._insert_done()
            logger.debug(f"Completed insert operations for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error in _insert_done for collection '{collection_name}': {e}")

    async def _query_done(self, collection_name: str):
        """
        Handle post-query operations.

        Args:
            collection_name (str): The name of the collection.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance._query_done()
            logger.debug(f"Completed query operations for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error in _query_done for collection '{collection_name}': {e}")

    async def _delete_by_entity_done(self, collection_name: str):
        """
        Handle post-deletion operations.

        Args:
            collection_name (str): The name of the collection.
        """
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance._delete_by_entity_done()
            logger.debug(f"Completed delete operations for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error in _delete_by_entity_done for collection '{collection_name}': {e}")

    # Example usage
    if __name__ == "__main__":
        import sys

        async def main():
            # Initialize LightRAGClient
            client = LightRAGClient(
                working_dir="./lightrag_cache",
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                graph_storage="NetworkXStorage",
                log_level=DEBUG,
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
                tiktoken_model_name="gpt-4o-mini",
                entity_extract_max_gleaning=1,
                entity_summary_to_max_tokens=500,
                node_embedding_algorithm="node2vec",
                node2vec_params={
                    "dimensions": 1536,
                    "num_walks": 10,
                    "walk_length": 40,
                    "window_size": 2,
                    "iterations": 3,
                    "random_seed": 3,
                },
                embedding_batch_num=32,
                embedding_func_max_async=16,
                llm_model_max_token_size=32768,
                llm_model_max_async=16,
                llm_model_kwargs={},
                vector_db_storage_cls_kwargs={},
                enable_llm_cache=True,
                addon_params={},
                convert_response_to_json_func=convert_response_to_json,
            )

            # Example insert
            documents = [
                "OpenAI develops artificial intelligence technologies.",
                "LightRAG integrates various storage backends for efficient retrieval.",
                "GPT-4o-mini is a language model used for generating responses."
            ]
            success = client.insert(items=documents, collection_name="default_collection")
            if success:
                print("Documents inserted successfully.")
            else:
                print("Failed to insert documents.")

            # Example query with filter
            query = "Tell me about OpenAI and LightRAG."
            filter_criteria = {"excluded_ids": ["doc-12345"]}  # Example filter
            response = client.query(
                query=query,
                collection_name="default_collection",
                param=QueryParam(mode="hybrid"),
                filter=filter_criteria
            )
            if isinstance(response, QueryResult):
                print("Query Response IDs:")
                print(response.ids)
            elif isinstance(response, str):
                print("Query Response:")
                print(response)
            else:
                print("No response for the query.")

            # Example search with vectors and limit
            search_query = "OpenAI"
            # Example vector: random vector for demonstration; replace with actual vectors
            search_vectors = [np.random.rand(1536)]
            search_limit = 5  # Example limit; adjust as needed
            search_response = client.search(
                query=search_query,
                collection_name="default_collection",
                vectors=search_vectors,
                limit=search_limit
            )
            if isinstance(search_response, QueryResult):
                print("Search Response IDs after similarity filtering and limit:")
                print(search_response.ids)
            else:
                print("No search results found.")

        # Run the main function
        asyncio.run(main())

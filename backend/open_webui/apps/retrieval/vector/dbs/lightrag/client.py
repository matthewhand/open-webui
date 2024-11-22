# client.py

import os
import asyncio
import threading  # Ensure threading is imported
import uuid
import json
import logging
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field, fields
from datetime import datetime
from .utils import (
    QueryResult,
    EmbeddingFunc,
    truncate_vector,
    truncate_vectors_for_logging,
    convert_response_to_json,
)
from .llm import wrapped_openai_embedding
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

# Import the custom client and storage classes
from .custom_lightrag import CustomLightRAGClient
from .custom_storage import CustomNanoVectorDBStorage

# Initialize logger
logger = logging.getLogger("lightrag_client")
logger.setLevel(logging.DEBUG)

# Custom Formatter to handle missing 'request_id'
class RequestIDFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        return super().format(record)

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = RequestIDFormatter(
    '%(asctime)s - %(levelname)s - [Request ID: %(request_id)s] - %(message)s'
)
console_handler.setFormatter(formatter)

# Avoid adding multiple handlers if they already exist
if not logger.hasHandlers():
    logger.addHandler(console_handler)
else:
    # Clear existing handlers and add the configured one
    logger.handlers.clear()
    logger.addHandler(console_handler)

# AsyncRunner to run coroutines in synchronous methods
class AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        logger.debug("AsyncRunner initialized with a new event loop.", extra={'request_id': 'N/A'})

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        logger.debug("AsyncRunner event loop started.", extra={'request_id': 'N/A'})

    def run(self, coro, request_id: str):
        """
        Submit a coroutine to the event loop and wait for its result.

        Args:
            coro: The coroutine to run.
            request_id: Unique identifier for the request.

        Returns:
            The result of the coroutine.
        """
        logger.debug(f"Running coroutine: {coro}", extra={'request_id': request_id})
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            result = future.result()
            logger.debug(f"Coroutine result: {result}", extra={'request_id': request_id})
            return result
        except Exception as e:
            logger.error(f"Exception during coroutine execution: {e}", extra={'request_id': request_id})
            raise

    def shutdown(self):
        """
        Shutdown the event loop and the background thread.
        """
        logger.debug("Shutting down AsyncRunner's event loop.", extra={'request_id': 'N/A'})
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        logger.info("AsyncRunner shutdown successfully.", extra={'request_id': 'N/A'})

@dataclass
class LightRAGClient:
    """
    Client class to interact with LightRAG instances for various operations like insert, search, and vector searches.
    """
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    log_level: int = field(default=logging.DEBUG)  # Default to DEBUG for detailed logs

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
        func=wrapped_openai_embedding  # Use the wrapper function
    ))
    llm_model_func: Callable = field(default_factory=lambda: gpt_4o_mini_complete)

    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: Dict = field(default_factory=dict)

    # Storage
    enable_llm_cache: bool = True

    # Extension
    addon_params: Dict = field(default_factory=dict)
    convert_response_to_json_func: Callable = convert_response_to_json

    # Storage Instances (Initialized in __post_init__)
    collections: Dict[str, 'LightRAG'] = field(default_factory=dict, init=False)

    # AsyncRunner instance
    async_runner: AsyncRunner = field(default_factory=AsyncRunner, init=False)

    # Initialize the custom client
    custom_client: CustomLightRAGClient = field(default_factory=CustomLightRAGClient, init=False)

    def __post_init__(self):
        """
        Post-initialization to set up logging and initialize the collections dictionary.
        """
        logger.debug(f"Initializing LightRAGClient with working_dir: '{self.working_dir}'", extra={'request_id': 'N/A'})
        logger.debug(f"LightRAGClient parameters: {self._get_initial_config()}", extra={'request_id': 'N/A'})

        # Initialize collections dict via custom client
        self.collections = self.custom_client.lightrag_instance.collections

    def _get_initial_config(self) -> Dict[str, Union[str, int, Dict, bool]]:
        """
        Retrieve initial configuration excluding non-serializable fields.

        Returns:
            Dict[str, Union[str, int, Dict, bool]]: Configuration dictionary.
        """
        exclude_fields = {
            'async_runner',
            'embedding_func',
            'llm_model_func',
            'convert_response_to_json_func',
            'collections',
            'custom_client'
        }
        config_dict = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude_fields}
        return config_dict

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        exists = collection_name in self.collections
        logger.debug(f"Checking existence of collection '{collection_name}': {exists}", extra={'request_id': 'N/A'})
        return exists

    def get_lightRAG_instance(self, collection_name: str) -> 'LightRAG':
        """
        Retrieve or initialize a LightRAG instance for the specified collection.
        """
        return self.custom_client.get_lightRAG_instance(collection_name)

    # --------------------- Asynchronous Methods ---------------------

    async def insert_async(self, items: Union[str, List[str], Dict], *, collection_name: str = "default") -> bool:
        """
        Asynchronously insert documents into the specified collection.

        Args:
            items (Union[str, List[str], Dict]): The text(s) to insert.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        return await self.custom_client.insert_async(items=items, collection_name=collection_name)

    # --------------------- Synchronous Methods ---------------------

    def insert(
        self,
        items: Union[str, List[str], Dict],
        *,
        collection_name: str = "default"
    ) -> bool:
        """
        Synchronously insert documents into the specified collection.

        Args:
            items (Union[str, List[str], Dict]): The text(s) to insert.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Calling synchronous insert with items: {items}, collection_name: '{collection_name}'", extra={'request_id': request_id})
        try:
            result = self.async_runner.run(
                self.insert_async(items=items, collection_name=collection_name),
                request_id
            )
            logger.debug(f"Synchronous insert result: {result}", extra={'request_id': request_id})
            return result
        except Exception as e:
            logger.error(f"Runtime error during synchronous insert: {e}", extra={'request_id': request_id})
            return False

    # --------------------- Query Methods ---------------------

    async def query_async(
        self,
        query: str = "",
        *,
        collection_name: str = "default",
        param: QueryParam = QueryParam(),
        filter: Optional[Dict] = None
    ) -> Optional[QueryResult]:
        """
        Asynchronously query the specified collection.

        Args:
            query (str, optional): The query string. Defaults to an empty string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            param (QueryParam, optional): Query parameters. Defaults to QueryParam().
            filter (Optional[Dict], optional): Filter criteria for the query. Defaults to None.

        Returns:
            Optional[QueryResult]: The query result or an error message encapsulated in QueryResult.
        """
        return await self.custom_client.query_async(query=query, collection_name=collection_name, param=param, filter=filter)

    def query(
        self,
        query: str = "",
        *,
        collection_name: str = "default",
        param: QueryParam = QueryParam(),
        filter: Optional[Dict] = None
    ) -> Optional[QueryResult]:
        """
        Synchronously query the specified collection.

        Args:
            query (str, optional): The query string. Defaults to an empty string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            param (QueryParam, optional): Query parameters. Defaults to QueryParam().
            filter (Optional[Dict], optional): Filter criteria for the query. Defaults to None.

        Returns:
            Optional[QueryResult]: The query result or an error message encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Calling synchronous query with query: '{query}', collection_name: '{collection_name}', param: {param}, filter: {filter}", extra={'request_id': request_id})
        try:
            result = self.async_runner.run(
                self.query_async(query=query, collection_name=collection_name, param=param, filter=filter),
                request_id
            )
            logger.debug(f"Synchronous query result: {result}", extra={'request_id': request_id})
            return result
        except Exception as e:
            logger.error(f"Runtime error during synchronous query: {e}", extra={'request_id': request_id})
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Runtime error during query.")

    # --------------------- Vector Search Methods ---------------------

    async def vector_search_async(
        self,
        vectors: List[Union[List[float], list]],
        *,
        collection_name: str = "default",
        limit: Optional[int] = 5
    ) -> Optional[QueryResult]:
        """
        Asynchronously perform vector search on the specified collection using LightRAG's vector storage.

        Args:
            vectors (List[Union[List[float], list]]): List of query vectors.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            limit (Optional[int], optional): Maximum number of results to return per vector. Defaults to 5.

        Returns:
            Optional[QueryResult]: The search result containing document IDs, embeddings, metadatas, documents, distances, or an error message encapsulated in QueryResult.
        """
        return await self.custom_client.vector_search_async(vectors=vectors, collection_name=collection_name, limit=limit)

    def vector_search(
        self,
        vectors: List[Union[List[float], list]],
        *,
        collection_name: str = "default",
        limit: Optional[int] = 5
    ) -> Optional[QueryResult]:
        """
        Synchronously perform vector search on the specified collection.

        Args:
            vectors (List[Union[List[float], list]]): List of query vectors.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            limit (Optional[int], optional): Maximum number of results to return per vector. Defaults to 5.

        Returns:
            Optional[QueryResult]: The search result containing document IDs, embeddings, metadatas, documents, distances, or an error message encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        truncated_vectors = self._truncate_vectors_for_logging(vectors)
        logger.debug(f"Calling synchronous vector_search with vectors: {truncated_vectors}, collection_name: '{collection_name}', limit: {limit}", extra={'request_id': request_id})
        try:
            result = self.async_runner.run(
                self.vector_search_async(vectors=vectors, collection_name=collection_name, limit=limit),
                request_id
            )
            logger.debug(f"Synchronous vector_search result: {result}", extra={'request_id': request_id})
            return result
        except Exception as e:
            logger.error(f"Runtime error during synchronous vector search: {e}", extra={'request_id': request_id})
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Runtime error during vector search.")

    # --------------------- Unified Search Method ---------------------

    def search(
        self,
        query: Optional[str] = None,
        *,
        collection_name: str = "default",
        mode: Optional[str] = None,
        vectors: Optional[List[List[float]]] = None,
        limit: Optional[int] = 5
    ) -> Optional[QueryResult]:
        """
        Unified search method to handle both string-based and vector-based searches.

        Args:
            query (Optional[str], optional): The query string. Defaults to None.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            mode (Optional[str], optional): The search mode for query-based searches. Defaults to None.
            vectors (Optional[List[List[float]]], optional): List of query vectors for similarity search. Defaults to None.
            limit (Optional[int], optional): Maximum number of results to return per vector. Defaults to 5.

        Returns:
            Optional[QueryResult]: The search result.
        """
        request_id = str(uuid.uuid4())
        if vectors:
            logger.debug("Performing vector-based search.", extra={'request_id': request_id})
            return self.vector_search(vectors=vectors, collection_name=collection_name, limit=limit)
        else:
            logger.debug("Performing string-based query.", extra={'request_id': request_id})
            return self.query(query=query, collection_name=collection_name, param=QueryParam(), filter=None)

    # --------------------- Deletion Methods ---------------------

    async def delete_by_entity_async(self, entity_name: str, *, collection_name: str = "default") -> bool:
        """
        Asynchronously delete an entity and its relationships from the specified collection.

        Args:
            entity_name (str): The name of the entity to delete.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        return await self.custom_client.delete_by_entity_async(entity_name, collection_name=collection_name)

    def delete_by_entity(self, entity_name: str, *, collection_name: str = "default") -> bool:
        """
        Synchronously delete an entity and its relationships from the specified collection.

        Args:
            entity_name (str): The name of the entity to delete.
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Calling synchronous delete_by_entity with entity_name: '{entity_name}', collection_name: '{collection_name}'", extra={'request_id': request_id})
        try:
            result = self.async_runner.run(
                self.delete_by_entity_async(entity_name, collection_name=collection_name),
                request_id
            )
            logger.debug(f"Synchronous delete_by_entity result: {result}", extra={'request_id': request_id})
            return result
        except Exception as e:
            logger.error(f"Runtime error during synchronous deletion: {e}", extra={'request_id': request_id})
            return False

    # --------------------- Helper Methods ---------------------

    async def _query_done(self, collection_name: str):
        """
        Handle post-query operations.

        Args:
            collection_name (str): The name of the collection.
        """
        await self.custom_client._query_done(collection_name)

    async def _insert_done(self, collection_name: str):
        """
        Handle post-insertion operations.

        Args:
            collection_name (str): The name of the collection.
        """
        await self.custom_client._insert_done(collection_name)

    async def _delete_by_entity_done(self, collection_name: str):
        """
        Handle post-deletion operations.

        Args:
            collection_name (str): The name of the collection.
        """
        await self.custom_client._delete_by_entity_done(collection_name)

    def _truncate_vector(self, vector: Union[List[float], list], max_elements: int = 3) -> str:
        """
        Truncate a vector to the first 'max_elements' elements for logging purposes.

        Args:
            vector (Union[List[float], list]): The vector to truncate.
            max_elements (int, optional): Number of elements to keep. Defaults to 3.

        Returns:
            str: Truncated vector as a string.
        """
        return truncate_vector(vector, max_elements)

    def _truncate_vectors_for_logging(self, vectors: Optional[List[Union[List[float], list]]], max_elements: int = 3) -> Optional[List[str]]:
        """
        Truncate all vectors in a list for logging purposes.

        Args:
            vectors (Optional[List[Union[List[float], list]]]): List of vectors to truncate.
            max_elements (int, optional): Number of elements to keep in each vector. Defaults to 3.

        Returns:
            Optional[List[str]]: List of truncated vectors as strings.
        """
        return truncate_vectors_for_logging(vectors, max_elements)

    # --------------------- Shutdown Method ---------------------

    def shutdown(self):
        """
        Shutdown the AsyncRunner's event loop.
        """
        request_id = str(uuid.uuid4())
        logger.debug("Shutting down LightRAGClient.", extra={'request_id': request_id})
        self.async_runner.shutdown()
        logger.info("LightRAGClient shutdown successfully.", extra={'request_id': request_id})

    # --------------------- Example Usage ---------------------
    # Inside LightRAGClient.run_example()

    def run_example(self):
        """
        Example usage of the LightRAGClient. This method is called only when the RUN_LIGHTRAG_MAIN environment variable is set to "1".
        """
        async def main():
            request_id = str(uuid.uuid4())
            logger.debug("Starting run_example.", extra={'request_id': request_id})

            # Example insert with a list of strings
            documents = [
                "OpenAI develops artificial intelligence technologies.",
                "LightRAG integrates various storage backends for efficient retrieval.",
                "GPT-4o-mini is a language model used for generating responses."
            ]
            logger.debug(f"Starting document insertion with documents: {documents}", extra={'request_id': request_id})
            success = await self.insert_async(items=documents, collection_name="default_collection")
            if success:
                print("Documents inserted successfully.")
                logger.info("Documents inserted successfully.", extra={'request_id': request_id})
            else:
                print("Failed to insert documents.")
                logger.error("Failed to insert documents.", extra={'request_id': request_id})

            # Example insert with a dictionary
            documents_dict = {
                "doc1": "OpenAI develops artificial intelligence technologies.",
                "doc2": "LightRAG integrates various storage backends for efficient retrieval.",
                "doc3": "GPT-4o-mini is a language model used for generating responses."
            }
            logger.debug(f"Starting document insertion from dictionary: {documents_dict}", extra={'request_id': request_id})
            success = await self.insert_async(items=documents_dict, collection_name="default_collection")
            if success:
                print("Documents from dictionary inserted successfully.")
                logger.info("Documents from dictionary inserted successfully.", extra={'request_id': request_id})
            else:
                print("Failed to insert documents from dictionary.")
                logger.error("Failed to insert documents from dictionary.", extra={'request_id': request_id})

            # Compute embeddings for inserted documents
            all_documents = documents + list(documents_dict.values())
            embeddings = await self.embedding_func(all_documents)  # Await the coroutine
            logger.debug("Computed embeddings for inserted documents.", extra={'request_id': request_id})

            # Verify embedding dimensions
            for idx, emb in enumerate(embeddings):
                if len(emb) != self.embedding_func.embedding_dim:
                    logger.error(f"Embedding for document {idx + 1} has incorrect dimensions: {len(emb)}. Expected: {self.embedding_func.embedding_dim}.", extra={'request_id': request_id})
                    print(f"Error: Embedding for document {idx + 1} has incorrect dimensions.")
                    return
            logger.debug("All embeddings have correct dimensions.", extra={'request_id': request_id})

            # Use the first embedding for querying
            query_embedding = embeddings[0]
            truncated_query_embedding = self._truncate_vector(query_embedding)
            logger.debug(f"Using embedding of first document for query: {truncated_query_embedding}", extra={'request_id': request_id})

            # Example vector search using the embedding of the first document
            search_vectors = [query_embedding]
            search_limit = 5  # Example limit; adjust as needed
            truncated_search_vectors = self._truncate_vectors_for_logging(search_vectors)
            logger.debug(f"Starting vector search with vectors: {truncated_search_vectors}, limit: {search_limit}", extra={'request_id': request_id})
            vector_search_response = await self.vector_search_async(
                vectors=search_vectors,
                collection_name="default_collection",
                limit=search_limit
            )
            if isinstance(vector_search_response, QueryResult):
                if vector_search_response.error:
                    print(f"Vector Search Error: {vector_search_response.error}")
                    logger.error(f"Vector Search Error: {vector_search_response.error}", extra={'request_id': request_id})
                else:
                    print("Vector Search Response IDs:")
                    print(vector_search_response.ids)
                    print("Vector Search Response Documents:")
                    print(vector_search_response.documents)
                    logger.info(f"Vector Search Response IDs: {vector_search_response.ids}", extra={'request_id': request_id})
            else:
                print("No vector search results found.")
                logger.warning("No vector search results found.", extra={'request_id': request_id})

            # Perform a Controlled Test: Insert and Search a Known Document
            logger.debug("Starting controlled test case.", extra={'request_id': request_id})
            known_document = "This is a test document for LightRAG."
            insert_success = await self.insert_async(items=known_document, collection_name="controlled_test_collection")
            if insert_success:
                logger.info("Controlled Test: Document inserted successfully.", extra={'request_id': request_id})
                # Compute embedding
                known_embedding = await self.embedding_func([known_document])  # Await the coroutine
                known_embedding = known_embedding[0]  # Extract the embedding from the list
                truncated_known_embedding = self._truncate_vector(known_embedding)
                logger.debug(f"Controlled Test: Computed embedding for known document: {truncated_known_embedding}", extra={'request_id': request_id})
                # Perform search with the same embedding
                search_result = await self.vector_search_async(
                    vectors=[known_embedding],
                    collection_name="controlled_test_collection",
                    limit=1
                )
                if isinstance(search_result, QueryResult):
                    if search_result.error:
                        print(f"Controlled Test Vector Search Error: {search_result.error}")
                        logger.error(f"Controlled Test Vector Search Error: {search_result.error}", extra={'request_id': request_id})
                    else:
                        print("Controlled Test Vector Search Response IDs:")
                        print(search_result.ids)
                        print("Controlled Test Vector Search Documents:")
                        print(search_result.documents)
                        logger.info(f"Controlled Test Vector Search Response IDs: {search_result.ids}", extra={'request_id': request_id})
                else:
                    print("Controlled Test: No vector search results found.")
                    logger.warning("Controlled Test: No vector search results found.", extra={'request_id': request_id})
            else:
                logger.error("Controlled Test: Failed to insert document.", extra={'request_id': request_id})

            # Shutdown the client
            self.shutdown()
            logger.info("LightRAGClient shutdown successfully.", extra={'request_id': request_id})

        asyncio.run(main())

    # --------------------- Additional Validation Methods ---------------------

    async def list_vectors_async(self, collection_name: str = "default") -> Optional[List[str]]:
        """
        Asynchronously list all vector IDs in the specified collection.

        Args:
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            Optional[List[str]]: List of vector IDs or None if an error occurs.
        """
        return await self.custom_client.list_vectors_async(collection_name=collection_name)

    async def list_documents_async(self, collection_name: str = "default") -> Optional[List[Dict[str, Any]]]:
        """
        Asynchronously list all documents in the specified collection.

        Args:
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            Optional[List[Dict[str, Any]]]: List of documents or None if an error occurs.
        """
        return await self.custom_client.list_documents_async(collection_name=collection_name)

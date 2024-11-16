# lightrag.py

import os
import asyncio
import uuid
import json
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, fields  # Ensure 'fields' is imported
from datetime import datetime
from logging import getLogger, StreamHandler, Formatter, DEBUG

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, openai_embedding
from lightrag.utils import EmbeddingFunc, convert_response_to_json
from pydantic import BaseModel
import threading  # Corrected import for Thread

# Initialize the main logger
logger = getLogger("lightrag_client")
logger.setLevel(DEBUG)  # Set to DEBUG for detailed logs
console_handler = StreamHandler()
console_handler.setLevel(DEBUG)
formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)
else:
    # Avoid adding multiple handlers
    logger.handlers.clear()
    logger.addHandler(console_handler)

# Define the QueryResult using Pydantic
class QueryResult(BaseModel):
    ids: List[List[str]] = []
    embeddings: Optional[List[List[float]]] = None
    metadatas: Optional[List[List[Dict[str, Any]]]] = None
    documents: Optional[List[List[str]]] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """
        Override dict() to handle numpy.ndarray serialization.
        """
        data = super().dict(**kwargs)
        if self.embeddings is not None:
            data['embeddings'] = [embedding for embedding in self.embeddings]
        return data

# AsyncRunner to run coroutines in synchronous methods
class AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)  # Corrected to use threading.Thread
        self.thread.start()
        logger.debug("AsyncRunner initialized with a new event loop.")

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        logger.debug("AsyncRunner event loop started.")

    def run(self, coro, request_id: str):
        """
        Submit a coroutine to the event loop and wait for its result.

        Args:
            coro: The coroutine to run.
            request_id: Unique identifier for the request.

        Returns:
            The result of the coroutine.
        """
        logger.debug(f"[Request ID: {request_id}] Running coroutine: {coro}")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            result = future.result()
            logger.debug(f"[Request ID: {request_id}] Coroutine result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Exception during coroutine execution: {e}")
            raise

    def shutdown(self):
        """
        Shutdown the event loop and the background thread.
        """
        logger.debug("Shutting down AsyncRunner's event loop.")
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        logger.info("AsyncRunner shutdown successfully.")

@dataclass
class LightRAGClient:
    """
    Client class to interact with LightRAG instances for various operations like insert, search, and query.
    """
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    log_level: int = field(default=DEBUG)  # Default to DEBUG for detailed logs

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
        func=lambda texts: openai_embedding(texts)
    ))
    llm_model_func: callable = field(default_factory=lambda: gpt_4o_mini_complete)

    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: Dict = field(default_factory=dict)

    # Storage
    enable_llm_cache: bool = True

    # Extension
    addon_params: Dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # Storage Instances (Initialized in __post_init__)
    collections: Dict[str, 'LightRAG'] = field(default_factory=dict, init=False)

    # AsyncRunner instance
    async_runner: AsyncRunner = field(default_factory=AsyncRunner, init=False)

    def __post_init__(self):
        """
        Post-initialization to set up logging and initialize the collections dictionary.
        """
        logger.debug(f"Initializing LightRAGClient with working_dir: '{self.working_dir}'")
        logger.debug(f"LightRAGClient parameters: {self._get_initial_config()}")

        # Initialize collections dict
        self.collections: Dict[str, 'LightRAG'] = {}

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
            'collections'
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
        logger.debug(f"Checking existence of collection '{collection_name}': {exists}")
        return exists

    def get_lightRAG_instance(self, collection_name: str) -> 'LightRAG':
        """
        Retrieve or initialize a LightRAG instance for the specified collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            LightRAG: The LightRAG instance for the collection.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Retrieving LightRAG instance for collection: '{collection_name}'")
        logger.debug(f"[Request ID: {request_id}] Working directory: '{self.working_dir}' (type: {type(self.working_dir)})")

        # Validate working_dir and collection_name
        if not isinstance(self.working_dir, str):
            logger.error(f"working_dir must be a string, got {type(self.working_dir)}: {self.working_dir}")
            raise TypeError(f"working_dir must be a string, got {type(self.working_dir)}")

        if not isinstance(collection_name, str):
            logger.error(f"collection_name must be a string, got {type(collection_name)}: {collection_name}")
            raise TypeError(f"collection_name must be a string, got {type(collection_name)}")

        if not collection_name.strip():
            logger.error("Collection name cannot be empty or whitespace.")
            raise ValueError("Collection name cannot be empty or whitespace.")

        collection_working_dir = os.path.join(self.working_dir, collection_name)
        logger.debug(f"[Request ID: {request_id}] Collection working directory: '{collection_working_dir}'")

        # Ensure the directory exists
        try:
            os.makedirs(collection_working_dir, exist_ok=True)
            logger.debug(f"[Request ID: {request_id}] Ensured existence of directory: '{collection_working_dir}'")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to create directory '{collection_working_dir}': {e}")
            raise

        if collection_name not in self.collections:
            try:
                lightrag_instance = LightRAG(
                    working_dir=collection_working_dir,
                    llm_model_func=self.llm_model_func,
                    embedding_func=self.embedding_func,
                    log_level=self.log_level,
                    chunk_token_size=self.chunk_token_size,
                    chunk_overlap_token_size=self.chunk_overlap_token_size,
                    tiktoken_model_name=self.tiktoken_model_name,
                    entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                    entity_summary_to_max_tokens=self.entity_summary_to_max_tokens,
                    node_embedding_algorithm=self.node_embedding_algorithm,
                    node2vec_params=self.node2vec_params,
                    embedding_batch_num=self.embedding_batch_num,
                    embedding_func_max_async=self.embedding_func_max_async,
                    llm_model_max_token_size=self.llm_model_max_token_size,
                    llm_model_max_async=self.llm_model_max_async,
                    llm_model_kwargs=self.llm_model_kwargs,
                    enable_llm_cache=self.enable_llm_cache,
                    addon_params=self.addon_params,
                    convert_response_to_json_func=self.convert_response_to_json_func,
                )
                self.collections[collection_name] = lightrag_instance
                logger.info(f"[Request ID: {request_id}] Initialized LightRAG instance for collection '{collection_name}'.")
            except Exception as e:
                logger.error(f"[Request ID: {request_id}] Failed to initialize LightRAG instance for collection '{collection_name}': {e}")
                raise
        else:
            logger.debug(f"[Request ID: {request_id}] Collection '{collection_name}' already exists.")

        return self.collections[collection_name]

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
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting insert_async with items: {items}, collection_name: '{collection_name}'")

        try:
            # Implementing guard to handle different types of 'items'
            if isinstance(items, dict):
                logger.debug(f"[Request ID: {request_id}] Detected dictionary in items: {items}")
                if not items:
                    logger.error(f"[Request ID: {request_id}] Empty dictionary provided for insertion.")
                    return False
                # Convert dictionary values to strings
                items = [str(value).strip() for value in items.values()]
                logger.debug(f"[Request ID: {request_id}] Converted items from dict: {items}")
            elif isinstance(items, str):
                logger.debug(f"[Request ID: {request_id}] Detected string in items: '{items}'")
                items = [items.strip()]
                logger.debug(f"[Request ID: {request_id}] Converted items to list: {items}")
            elif isinstance(items, list):
                logger.debug(f"[Request ID: {request_id}] Detected list in items: {items}")
                # Ensure all items in the list are strings
                items = [str(item).strip() for item in items]
                logger.debug(f"[Request ID: {request_id}] Sanitized items: {items}")
            else:
                logger.error(f"[Request ID: {request_id}] Unsupported type for items: {type(items)}. Expected str, list of str, or dict.")
                return False

            # Further sanitize items by removing empty strings
            original_length = len(items)
            items = [item for item in items if item]
            sanitized_length = len(items)
            if sanitized_length < original_length:
                logger.warning(f"[Request ID: {request_id}] Removed {original_length - sanitized_length} empty or whitespace-only items.")

            if not items:
                logger.error(f"[Request ID: {request_id}] No valid documents to insert after sanitization.")
                return False

            logger.debug(f"[Request ID: {request_id}] Inserting {len(items)} documents into collection '{collection_name}'.")
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            result = await lightrag_instance.ainsert(items)
            logger.debug(f"[Request ID: {request_id}] Insertion result: {result}")
            return result
        except AttributeError as e:
            logger.error(f"[Request ID: {request_id}] Attribute error during insertion: {e}")
            return False
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to insert documents into collection '{collection_name}': {e}")
            return False
        finally:
            logger.debug(f"[Request ID: {request_id}] Completed insert_async.")
            if self.has_collection(collection_name):
                await self._insert_done(collection_name)

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
        logger.debug(f"[Request ID: {request_id}] Calling synchronous insert with items: {items}, collection_name: '{collection_name}'")
        try:
            result = self.async_runner.run(
                self.insert_async(items=items, collection_name=collection_name),
                request_id
            )
            logger.debug(f"[Request ID: {request_id}] Synchronous insert result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Runtime error during synchronous insert: {e}")
            return False

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
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting query_async with query: '{query}', collection_name: '{collection_name}', param: {param}, filter: {filter}")

        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)

            # Validate query
            if not isinstance(query, str):
                error_msg = "Query must be a string."
                logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(query)}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)

            if not query.strip():
                error_msg = "Query string cannot be empty."
                logger.warning(f"[Request ID: {request_id}] {error_msg}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)

            if filter:
                # Validate filter structure
                if not isinstance(filter, dict):
                    error_msg = "Filter must be a dictionary."
                    logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(filter)}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)
                if 'excluded_ids' in filter and not isinstance(filter['excluded_ids'], list):
                    error_msg = "'excluded_ids' in filter must be a list."
                    logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(filter['excluded_ids'])}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)

                logger.debug(f"[Request ID: {request_id}] Performing query with filter: {filter}")
                response = await lightrag_instance.aquery(query, param)
                logger.debug(f"[Request ID: {request_id}] Received response from aquery with filter: {response} (type: {type(response)})")

                # Check if response has 'ids' attribute
                if hasattr(response, 'ids') and response.ids and len(response.ids) > 0:
                    excluded_ids = filter.get('excluded_ids', [])
                    logger.debug(f"[Request ID: {request_id}] Excluded IDs: {excluded_ids}")

                    # Filter out excluded IDs
                    filtered_ids = [doc_id for doc_id in response.ids[0] if doc_id not in excluded_ids]
                    logger.debug(f"[Request ID: {request_id}] Filtered IDs count: {len(filtered_ids)} out of {len(response.ids[0])}")

                    return QueryResult(ids=[filtered_ids], embeddings=response.embeddings, metadatas=response.metadatas, documents=response.documents, error=None)
                else:
                    error_msg = "No documents found for the given query."
                    logger.error(f"[Request ID: {request_id}] {error_msg}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)
            else:
                # Perform a standard generative query
                logger.debug(f"[Request ID: {request_id}] Performing standard query without filter.")
                response = await lightrag_instance.aquery(query, param)
                logger.debug(f"[Request ID: {request_id}] Received query response: {response} (type: {type(response)})")

                # Check if response has 'ids' attribute
                if hasattr(response, 'ids') and response.ids and len(response.ids) > 0:
                    logger.debug(f"[Request ID: {request_id}] Returning QueryResult with IDs: {response.ids}")
                    return QueryResult(ids=response.ids, embeddings=response.embeddings, metadatas=response.metadatas, documents=response.documents, error=None)
                elif isinstance(response, str):
                    try:
                        json_response = json.loads(response)
                        if 'error' in json_response:
                            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=json_response['error'])
                        else:
                            error_msg = "LLM returned an unexpected string response without an error key."
                            logger.error(f"[Request ID: {request_id}] {error_msg} Content: {response}")
                            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)
                    except json.JSONDecodeError:
                        error_msg = "Unexpected string response from query."
                        logger.error(f"[Request ID: {request_id}] {error_msg} Content: {response}")
                        return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)
                else:
                    error_msg = "Unexpected response format from query."
                    logger.error(f"[Request ID: {request_id}] {error_msg} Response type: {type(response)}. Content: {response}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Query failed for '{query}' with error: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Internal server error during query processing.")
        finally:
            logger.debug(f"[Request ID: {request_id}] Completed query_async.")
            await self._query_done(collection_name)

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
        logger.debug(f"[Request ID: {request_id}] Calling synchronous query with query: '{query}', collection_name: '{collection_name}', param: {param}, filter: {filter}")
        try:
            result = self.async_runner.run(
                self.query_async(query=query, collection_name=collection_name, param=param, filter=filter),
                request_id
            )
            logger.debug(f"[Request ID: {request_id}] Synchronous query result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Runtime error during synchronous query: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Runtime error during query.")

    # --------------------- Vector Search Methods ---------------------

    async def vector_search_async(
        self,
        vectors: List[np.ndarray],
        *,
        collection_name: str = "default",
        limit: Optional[int] = None
    ) -> Optional[QueryResult]:
        """
        Asynchronously perform vector search on the specified collection.

        Args:
            vectors (List[np.ndarray]): List of query vectors.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            limit (Optional[int], optional): Maximum number of results to return per vector. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result containing document IDs, embeddings, metadatas, documents, or an error message encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting vector_search_async with {len(vectors)} vectors, limit: {limit}, collection_name: '{collection_name}'")

        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            # Assuming LightRAG has a 'vector_storage' attribute
            vector_storage = getattr(lightrag_instance, 'vector_storage', None)
            if vector_storage is None:
                logger.error(f"[Request ID: {request_id}] LightRAG instance does not have 'vector_storage' attribute.")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Vector storage not initialized.")

            # Initialize lists to collect results
            all_doc_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []

            for idx, vector in enumerate(vectors):
                logger.debug(f"[Request ID: {request_id}] Processing vector {idx + 1}/{len(vectors)}")
                try:
                    # Perform similarity search using the 'query' method
                    results = await vector_storage.query(query=vector.tolist(), top_k=limit or 5)
                    logger.debug(f"[Request ID: {request_id}] Retrieved {len(results)} results for vector {idx + 1}")

                    if results:
                        doc_ids = [result['id'] for result in results]
                        embeddings = [result.get('embedding', []) for result in results]
                        metadatas = [result.get('metadata', {}) for result in results]
                        documents = [result.get('document', "") for result in results]

                        all_doc_ids.append(doc_ids)
                        all_embeddings.append(embeddings)
                        all_metadatas.append(metadatas)
                        all_documents.append(documents)
                    else:
                        logger.warning(f"[Request ID: {request_id}] No results found for vector {idx + 1}")
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                except Exception as e:
                    logger.error(f"[Request ID: {request_id}] Error during vector search for vector {idx + 1}: {e}")
                    all_doc_ids.append([])
                    all_embeddings.append([])
                    all_metadatas.append([])
                    all_documents.append([])

            logger.debug(f"[Request ID: {request_id}] Vector search completed for all vectors.")
            return QueryResult(
                ids=all_doc_ids,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                documents=all_documents,
                error=None
            )
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Vector search failed with error: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Internal server error during vector search.")
        finally:
            logger.debug(f"[Request ID: {request_id}] Completed vector_search_async.")
            await self._query_done(collection_name)

    def vector_search(
        self,
        vectors: List[np.ndarray],
        *,
        collection_name: str = "default",
        limit: Optional[int] = None
    ) -> Optional[QueryResult]:
        """
        Synchronously perform vector search on the specified collection.

        Args:
            vectors (List[np.ndarray]): List of query vectors.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            limit (Optional[int], optional): Maximum number of results to return per vector. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result containing document IDs, embeddings, metadatas, documents, or an error message encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Calling synchronous vector_search with {len(vectors)} vectors, collection_name: '{collection_name}', limit: {limit}")
        try:
            result = self.async_runner.run(
                self.vector_search_async(vectors, collection_name=collection_name, limit=limit),
                request_id
            )
            logger.debug(f"[Request ID: {request_id}] Synchronous vector_search result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Runtime error during synchronous vector search: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Runtime error during vector search.")

    async def search_async(
        self,
        query: Optional[str] = None,
        *,
        collection_name: str = "default",
        vectors: Optional[List[np.ndarray]] = None,
        limit: Optional[int] = None
    ) -> Optional[QueryResult]:
        """
        Asynchronous search handler.

        Redirects to `vector_search_async` if vectors are provided, or performs a query-based search otherwise.

        Args:
            query (Optional[str], optional): The query string. If None, a dummy value is used.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            vectors (Optional[List[np.ndarray]], optional): Vectors to perform similarity search. Defaults to None.
            limit (Optional[int], optional): Maximum number of results to return per vector or query. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result containing document IDs, embeddings, metadatas, documents, or an error message encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting search_async with query: '{query}', vectors: {vectors}, limit: {limit}, collection_name: '{collection_name}'")

        try:
            if vectors:
                logger.debug(f"[Request ID: {request_id}] Vectors provided. Redirecting to vector_search_async.")
                return await self.vector_search_async(vectors, collection_name=collection_name, limit=limit)
            elif query:
                logger.debug(f"[Request ID: {request_id}] Query provided. Redirecting to query_async.")
                return await self.query_async(query=query, collection_name=collection_name)
            else:
                error_msg = "Either `query` or `vectors` must be provided for search."
                logger.error(f"[Request ID: {request_id}] {error_msg}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error=error_msg)
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Search failed with error: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Internal server error during search.")
        finally:
            logger.debug(f"[Request ID: {request_id}] Completed search_async.")
            await self._query_done(collection_name)

    def search(
        self,
        query: Optional[str] = None,
        *,
        collection_name: str = "default",
        vectors: Optional[List[np.ndarray]] = None,
        limit: Optional[int] = None
    ) -> Optional[QueryResult]:
        """
        Synchronously search the specified collection with optional vectors for similarity filtering and limit.
        Uses a dummy value for 'query' if not provided.

        Args:
            query (Optional[str], optional): The search query string. If None, a dummy value is used.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            vectors (Optional[List[np.ndarray]], optional): Vectors to perform similarity search. Defaults to None.
            limit (Optional[int], optional): Maximum number of results to return per vector or query. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        if query is None:
            logger.warning(f"[Request ID: {request_id}] No query provided to search. Using dummy value 'default_query'.")
            query = "default_query"  # Dummy value

        logger.debug(f"[Request ID: {request_id}] Calling synchronous search with query: '{query}', collection_name: '{collection_name}', vectors: {vectors}, limit: {limit}")
        try:
            result = self.async_runner.run(
                self.search_async(query=query, collection_name=collection_name, vectors=vectors, limit=limit),
                request_id
            )
            logger.debug(f"[Request ID: {request_id}] Synchronous search result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Runtime error during synchronous search: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, error="Runtime error during synchronous search.")

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
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting delete_by_entity_async with entity_name: '{entity_name}', collection_name: '{collection_name}'")

        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance.adelete_by_entity(entity_name)
            logger.debug(f"[Request ID: {request_id}] Successfully deleted entity '{entity_name}' from collection '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to delete entity '{entity_name}' from collection '{collection_name}': {e}")
            return False
        finally:
            logger.debug(f"[Request ID: {request_id}] Completed delete_by_entity_async.")
            await self._delete_by_entity_done(collection_name)

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
        logger.debug(f"[Request ID: {request_id}] Calling synchronous delete_by_entity with entity_name: '{entity_name}', collection_name: '{collection_name}'")
        try:
            result = self.async_runner.run(
                self.delete_by_entity_async(entity_name, collection_name=collection_name),
                request_id
            )
            logger.debug(f"[Request ID: {request_id}] Synchronous delete_by_entity result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Runtime error during synchronous deletion: {e}")
            return False

    # --------------------- Helper Methods ---------------------

    async def _query_done(self, collection_name: str):
        """
        Handle post-query operations.

        Args:
            collection_name (str): The name of the collection.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting _query_done for collection '{collection_name}'")
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance._query_done()
            logger.debug(f"[Request ID: {request_id}] Completed query operations for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Error in _query_done for collection '{collection_name}': {e}")

    async def _insert_done(self, collection_name: str):
        """
        Handle post-insertion operations.

        Args:
            collection_name (str): The name of the collection.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting _insert_done for collection '{collection_name}'")
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance._insert_done()
            logger.debug(f"[Request ID: {request_id}] Completed insert operations for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Error in _insert_done for collection '{collection_name}': {e}")

    async def _delete_by_entity_done(self, collection_name: str):
        """
        Handle post-deletion operations.

        Args:
            collection_name (str): The name of the collection.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting _delete_by_entity_done for collection '{collection_name}'")
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            await lightrag_instance._delete_by_entity_done()
            logger.debug(f"[Request ID: {request_id}] Completed delete operations for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Error in _delete_by_entity_done for collection '{collection_name}': {e}")

    # --------------------- Shutdown Method ---------------------

    def shutdown(self):
        """
        Shutdown the AsyncRunner's event loop.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Shutting down LightRAGClient.")
        self.async_runner.shutdown()
        logger.info(f"[Request ID: {request_id}] LightRAGClient shutdown successfully.")

    # --------------------- Example Usage ---------------------
    def run_example(self):
        """
        Example usage of the LightRAGClient. This method is called only when the RUN_LIGHTRAG_MAIN environment variable is set to "1".
        """
        async def main():
            request_id = str(uuid.uuid4())
            logger.debug(f"[Request ID: {request_id}] Starting run_example.")

            # Example insert with a list of strings
            documents = [
                "OpenAI develops artificial intelligence technologies.",
                "LightRAG integrates various storage backends for efficient retrieval.",
                "GPT-4o-mini is a language model used for generating responses."
            ]
            logger.debug(f"[Request ID: {request_id}] Starting document insertion with documents: {documents}")
            success = await self.insert_async(items=documents, collection_name="default_collection")
            if success:
                print("Documents inserted successfully.")
                logger.info(f"[Request ID: {request_id}] Documents inserted successfully.")
            else:
                print("Failed to insert documents.")
                logger.error(f"[Request ID: {request_id}] Failed to insert documents.")

            # Example insert with a dictionary
            documents_dict = {
                "doc1": "OpenAI develops artificial intelligence technologies.",
                "doc2": "LightRAG integrates various storage backends for efficient retrieval.",
                "doc3": "GPT-4o-mini is a language model used for generating responses."
            }
            logger.debug(f"[Request ID: {request_id}] Starting document insertion from dictionary: {documents_dict}")
            success = await self.insert_async(items=documents_dict, collection_name="default_collection")
            if success:
                print("Documents from dictionary inserted successfully.")
                logger.info(f"[Request ID: {request_id}] Documents from dictionary inserted successfully.")
            else:
                print("Failed to insert documents from dictionary.")
                logger.error(f"[Request ID: {request_id}] Failed to insert documents from dictionary.")

            # Example query with filter
            query = "Tell me about OpenAI and LightRAG."
            filter_criteria = {"excluded_ids": ["doc-12345"]}  # Example filter
            logger.debug(f"[Request ID: {request_id}] Starting query with filter: query='{query}', filter={filter_criteria}")
            response = await self.query_async(
                query=query,
                collection_name="default_collection",
                param=QueryParam(mode="hybrid"),
                filter=filter_criteria
            )
            if isinstance(response, QueryResult):
                if response.error:
                    print(f"Query Error: {response.error}")
                    logger.error(f"[Request ID: {request_id}] Query Error: {response.error}")
                else:
                    print("Query Response IDs:")
                    print(response.ids)
                    logger.info(f"[Request ID: {request_id}] Query Response IDs: {response.ids}")
            else:
                print("No response for the query.")
                logger.warning(f"[Request ID: {request_id}] No response received for the query.")

            # Example search with vectors and limit
            search_query = "OpenAI"
            # Example vector: random vector for demonstration; replace with actual vectors
            search_vectors = [np.random.rand(self.embedding_func.embedding_dim)]
            search_limit = 5  # Example limit; adjust as needed
            logger.debug(f"[Request ID: {request_id}] Starting search with query: '{search_query}', vectors: {search_vectors}, limit: {search_limit}")
            search_response = await self.search_async(
                query=search_query,
                collection_name="default_collection",
                vectors=search_vectors,
                limit=search_limit
            )
            if isinstance(search_response, QueryResult):
                if search_response.error:
                    print(f"Search Error: {search_response.error}")
                    logger.error(f"[Request ID: {request_id}] Search Error: {search_response.error}")
                else:
                    print("Search Response IDs after similarity filtering and limit:")
                    print(search_response.ids)
                    logger.info(f"[Request ID: {request_id}] Search Response IDs: {search_response.ids}")
            else:
                print("No search results found.")
                logger.warning(f"[Request ID: {request_id}] No search results found.")

            # Shutdown the client
            self.shutdown()
            logger.info(f"[Request ID: {request_id}] LightRAGClient shutdown successfully.")

        asyncio.run(main())

    # --------------------- Shutdown on Exit ---------------------

    def __del__(self):
        """
        Destructor to ensure that the AsyncRunner is properly shutdown.
        """
        try:
            self.shutdown()
        except Exception:
            pass

# --------------------- Conditional Main Execution ---------------------
if __name__ == "__main__":
    # Only run the example if the environment variable RUN_LIGHTRAG_MAIN is set to "1"
    if os.getenv("RUN_LIGHTRAG_MAIN") == "1":
        client = LightRAGClient()
        client.run_example()

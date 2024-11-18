# lightrag.py

import os
import asyncio
import uuid
import json
import numpy as np
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field, fields
from datetime import datetime
from logging import getLogger, StreamHandler, Formatter, DEBUG
import threading

# Importing necessary LightRAG modules
from backend.open_webui.apps.retrieval.vector.dbs.lightrag_client import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, openai_embedding

# Import helper functions and data models
from open_webui.apps.retrieval.vector.dbs.lightrag_utils import truncate_vector, truncate_vectors_for_logging, convert_response_to_json
from open_webui.apps.retrieval.vector.dbs.lightrag_utils import QueryResult, EmbeddingFunc

# Initialize the main logger
logger = getLogger("lightrag_client")
logger.setLevel(DEBUG)  # Set to DEBUG for detailed logs

# Configure console handler
console_handler = StreamHandler()
console_handler.setLevel(DEBUG)
formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
        func=openai_embedding  # Direct reference instead of a lambda
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
            await lightrag_instance.ainsert(items)
            logger.debug(f"[Request ID: {request_id}] Insertion succeeded.")
            return True
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
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting query_async with query: '{query}', collection_name: '{collection_name}', param: {param}, filter: {filter}")

        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)

            # Validate query
            if not isinstance(query, str):
                error_msg = "Query must be a string."
                logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(query)}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            if not query.strip():
                error_msg = "Query string cannot be empty."
                logger.warning(f"[Request ID: {request_id}] {error_msg}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            if filter:
                # Validate filter structure
                if not isinstance(filter, dict):
                    error_msg = "Filter must be a dictionary."
                    logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(filter)}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
                if 'excluded_ids' in filter and not isinstance(filter['excluded_ids'], list):
                    error_msg = "'excluded_ids' in filter must be a list."
                    logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(filter['excluded_ids'])}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

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

                    # Calculate distances if necessary; for now, use placeholders
                    distances = [0.0 for _ in filtered_ids]  # Placeholder distances

                    # Fetch documents and metadata for filtered IDs
                    documents = await lightrag_instance.afetch_documents(filtered_ids)
                    metadatas = await lightrag_instance.afetch_metadatas(filtered_ids)

                    return QueryResult(
                        ids=[filtered_ids],
                        embeddings=response.embeddings,
                        metadatas=metadatas,
                        documents=documents,
                        distances=[[d for d in distances]],  # Wrap in list to match structure
                        error=None
                    )
                else:
                    error_msg = "No documents found for the given query."
                    logger.error(f"[Request ID: {request_id}] {error_msg}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
            else:
                # Perform a standard generative query
                logger.debug(f"[Request ID: {request_id}] Performing standard query without filter.")
                response = await lightrag_instance.aquery(query, param)
                logger.debug(f"[Request ID: {request_id}] Received query response: {response} (type: {type(response)})")

                # Check if response has 'ids' attribute
                if hasattr(response, 'ids') and response.ids and len(response.ids) > 0:
                    logger.debug(f"[Request ID: {request_id}] Returning QueryResult with IDs: {response.ids}")
                    # Calculate distances if necessary; for now, use placeholders
                    distances = [[0.0 for _ in ids] for ids in response.ids]  # Placeholder distances

                    # Fetch documents and metadata for IDs
                    documents = await lightrag_instance.afetch_documents(response.ids[0])
                    metadatas = await lightrag_instance.afetch_metadatas(response.ids[0])

                    return QueryResult(
                        ids=response.ids,
                        embeddings=response.embeddings,
                        metadatas=metadatas,
                        documents=documents,
                        distances=distances,  # Include distances
                        error=None
                    )
                elif isinstance(response, str):
                    try:
                        json_response = json.loads(response)
                        if 'error' in json_response:
                            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=json_response['error'])
                        else:
                            error_msg = "LLM returned an unexpected string response without an error key."
                            logger.error(f"[Request ID: {request_id}] {error_msg} Content: {response}")
                            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
                    except json.JSONDecodeError:
                        error_msg = "Unexpected string response from query."
                        logger.error(f"[Request ID: {request_id}] {error_msg} Content: {response}")
                        return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
                else:
                    error_msg = "Unexpected response format from query."
                    logger.error(f"[Request ID: {request_id}] {error_msg} Response type: {type(response)}. Content: {response}")
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Query failed for '{query}' with error: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Internal server error during query processing.")
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
        request_id = str(uuid.uuid4())
        truncated_vectors = truncate_vectors_for_logging(vectors)
        logger.debug(f"[Request ID: {request_id}] Starting vector_search_async with vectors: {truncated_vectors}, limit: {limit}, collection_name: '{collection_name}'")

        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)

            # Access the appropriate vector storage; e.g., 'chunks_vdb'
            vector_storage = getattr(lightrag_instance, 'chunks_vdb', None)
            if vector_storage is None:
                error_msg = f"Vector storage 'chunks_vdb' not found in collection '{collection_name}'."
                logger.error(f"[Request ID: {request_id}] {error_msg}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            all_doc_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []
            all_distances = []

            for idx, vector in enumerate(vectors):
                truncated_vector = truncate_vector(vector)
                logger.debug(f"[Request ID: {request_id}] Processing vector {idx + 1}/{len(vectors)}: {truncated_vector}")
                try:
                    # Ensure the vector is in list format and has correct dimensions
                    if isinstance(vector, (list, tuple)):
                        query_vector = vector
                        logger.debug(f"[Request ID: {request_id}] Vector {idx + 1} is already a list.")
                    else:
                        logger.error(f"[Request ID: {request_id}] Unsupported vector type: {type(vector)}. Skipping this vector.")
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])
                        continue

                    # Validate embedding dimensions
                    if len(query_vector) != self.embedding_func.embedding_dim:
                        error_msg = f"Vector {idx + 1} has incorrect dimensions: {len(query_vector)}. Expected: {self.embedding_func.embedding_dim}."
                        logger.error(f"[Request ID: {request_id}] {error_msg}")
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])
                        continue

                    # Log the query vector for debugging (truncated)
                    logger.debug(f"[Request ID: {request_id}] Query Vector {idx + 1}: {truncated_vector}")

                    # Perform similarity search using the vector storage's query method
                    search_results = await vector_storage._client.query(
                        query=query_vector,
                        top_k=limit,
                        better_than_threshold=0.0  # Temporarily set to 0.0 to include all results
                    )
                    logger.debug(f"[Request ID: {request_id}] Retrieved {len(search_results)} results for vector {idx + 1}")

                    # Process search_results into QueryResult format
                    # Assuming search_results is a list of dicts with keys: 'id', 'embedding', 'metadata', 'document', 'distance'
                    doc_ids = [result['id'] for result in search_results]
                    embeddings = [result['embedding'] for result in search_results]
                    metadatas = [result['metadata'] for result in search_results]
                    documents = [result['document'] for result in search_results]
                    distances = [result['distance'] for result in search_results]

                    all_doc_ids.append(doc_ids)
                    all_embeddings.append(embeddings)
                    all_metadatas.append(metadatas)
                    all_documents.append(documents)
                    all_distances.append(distances)

                    logger.debug(f"[Request ID: {request_id}] Aggregated results for vector {idx + 1}: {doc_ids}")
                except Exception as e:
                    logger.error(f"[Request ID: {request_id}] Error during vector search for vector {idx + 1}: {e}")
                    all_doc_ids.append([])
                    all_embeddings.append([])
                    all_metadatas.append([])
                    all_documents.append([])
                    all_distances.append([])

            logger.debug(f"[Request ID: {request_id}] Vector search completed for all vectors.")
            return QueryResult(
                ids=all_doc_ids,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                documents=all_documents,
                distances=all_distances,  # Include distances
                error=None
            )
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Vector search failed with error: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Internal server error during vector search.")
        finally:
            logger.debug(f"[Request ID: {request_id}] Completed vector_search_async.")
            await self._query_done(collection_name)

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
        logger.debug(f"[Request ID: {request_id}] Calling synchronous vector_search with vectors: {truncated_vectors}, collection_name: '{collection_name}', limit: {limit}")
        try:
            result = self.async_runner.run(
                self.vector_search_async(vectors, collection_name=collection_name, limit=limit),
                request_id
            )
            logger.debug(f"[Request ID: {request_id}] Synchronous vector_search result: {result}")
            return result
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Runtime error during synchronous vector search: {e}")
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
        if vectors:
            return self.vector_search(vectors=vectors, collection_name=collection_name, limit=limit)
        else:
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

            # Compute embeddings for inserted documents
            all_documents = documents + list(documents_dict.values())
            embeddings = self.embedding_func.func(all_documents)
            logger.debug(f"[Request ID: {request_id}] Computed embeddings for inserted documents.")

            # Verify embedding dimensions
            for idx, emb in enumerate(embeddings):
                if len(emb) != self.embedding_func.embedding_dim:
                    logger.error(f"[Request ID: {request_id}] Embedding for document {idx + 1} has incorrect dimensions: {len(emb)}. Expected: {self.embedding_func.embedding_dim}.")
                    print(f"Error: Embedding for document {idx + 1} has incorrect dimensions.")
                    return
            logger.debug(f"[Request ID: {request_id}] All embeddings have correct dimensions.")

            # Use the first embedding for querying
            query_embedding = embeddings[0]
            truncated_query_embedding = self._truncate_vector(query_embedding)
            logger.debug(f"[Request ID: {request_id}] Using embedding of first document for query: {truncated_query_embedding}")

            # Example vector search using the embedding of the first document
            search_vectors = [query_embedding]
            search_limit = 5  # Example limit; adjust as needed
            truncated_search_vectors = self._truncate_vectors_for_logging(search_vectors)
            logger.debug(f"[Request ID: {request_id}] Starting vector search with vectors: {truncated_search_vectors}, limit: {search_limit}")
            vector_search_response = await self.vector_search_async(
                vectors=search_vectors,
                collection_name="default_collection",
                limit=search_limit
            )
            if isinstance(vector_search_response, QueryResult):
                if vector_search_response.error:
                    print(f"Vector Search Error: {vector_search_response.error}")
                    logger.error(f"[Request ID: {request_id}] Vector Search Error: {vector_search_response.error}")
                else:
                    print("Vector Search Response IDs:")
                    print(vector_search_response.ids)
                    print("Vector Search Response Documents:")
                    print(vector_search_response.documents)
                    logger.info(f"[Request ID: {request_id}] Vector Search Response IDs: {vector_search_response.ids}")
            else:
                print("No vector search results found.")
                logger.warning(f"[Request ID: {request_id}] No vector search results found.")

            # Perform a Controlled Test: Insert and Search a Known Document
            logger.debug(f"[Request ID: {request_id}] Starting controlled test case.")
            known_document = "This is a test document for LightRAG."
            insert_success = await self.insert_async(items=known_document, collection_name="controlled_test_collection")
            if insert_success:
                logger.info(f"[Request ID: {request_id}] Controlled Test: Document inserted successfully.")
                # Compute embedding
                known_embedding = self.embedding_func.func([known_document])[0]
                truncated_known_embedding = self._truncate_vector(known_embedding)
                logger.debug(f"[Request ID: {request_id}] Controlled Test: Computed embedding for known document: {truncated_known_embedding}")
                # Perform search with the same embedding
                search_result = await self.vector_search_async(
                    vectors=[known_embedding],
                    collection_name="controlled_test_collection",
                    limit=1
                )
                if isinstance(search_result, QueryResult):
                    if search_result.error:
                        print(f"Controlled Test Vector Search Error: {search_result.error}")
                        logger.error(f"[Request ID: {request_id}] Controlled Test Vector Search Error: {search_result.error}")
                    else:
                        print("Controlled Test Vector Search Response IDs:")
                        print(search_result.ids)
                        print("Controlled Test Vector Search Response Documents:")
                        print(search_result.documents)
                        logger.info(f"[Request ID: {request_id}] Controlled Test Vector Search Response IDs: {search_result.ids}")
                else:
                    print("Controlled Test: No vector search results found.")
                    logger.warning(f"[Request ID: {request_id}] Controlled Test: No vector search results found.")
            else:
                logger.error(f"[Request ID: {request_id}] Controlled Test: Failed to insert document.")

            # Shutdown the client
            self.shutdown()
            logger.info(f"[Request ID: {request_id}] LightRAGClient shutdown successfully.")

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
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting list_vectors_async for collection '{collection_name}'")
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            vector_storage = getattr(lightrag_instance, 'chunks_vdb', None)
            if vector_storage is None:
                logger.error(f"[Request ID: {request_id}] Vector storage 'chunks_vdb' not found in collection '{collection_name}'.")
                return None
            # Assuming the vector storage client has a method to list vectors
            all_vectors = await vector_storage._client.list_vectors()  # Adjust based on actual API
            logger.debug(f"[Request ID: {request_id}] Retrieved vectors: {all_vectors}")
            return all_vectors
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to list vectors: {e}")
            return None

    async def list_documents_async(self, collection_name: str = "default") -> Optional[List[Dict[str, Any]]]:
        """
        Asynchronously list all documents in the specified collection.

        Args:
            collection_name (str, optional): The name of the collection. Defaults to "default".

        Returns:
            Optional[List[Dict[str, Any]]]: List of documents or None if an error occurs.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting list_documents_async for collection '{collection_name}'")
        try:
            lightrag_instance = self.get_lightRAG_instance(collection_name)
            # Assuming the LightRAG instance has a method to list documents
            all_documents = await lightrag_instance.alist_documents()  # Adjust based on actual API
            logger.debug(f"[Request ID: {request_id}] Retrieved documents: {all_documents}")
            return all_documents
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to list documents: {e}")
            return None

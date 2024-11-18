# lightrag_search.py

import uuid
import json
from typing import Optional, List, Dict, Any, Union
from logging import getLogger, DEBUG

from backend.open_webui.apps.retrieval.vector.dbs.lightrag_client import LightRAG, QueryParam
from open_webui.apps.retrieval.vector.dbs.lightrag_utils import QueryResult, EmbeddingFunc

logger = getLogger("lightrag_search")
logger.setLevel(DEBUG)

class SearchHandler:
    """
    Handles search-related operations for LightRAGClient, including vector searches and querying mechanisms.
    """
    def __init__(self, client: 'LightRAGClient'):
        """
        Initialize the SearchHandler with a reference to the LightRAGClient.

        Args:
            client (LightRAGClient): The main LightRAGClient instance.
        """
        self.client = client
        logger.debug("SearchHandler initialized with LightRAGClient.")

    async def query_async(
        self,
        query: str = "",
        *,
        collection_name: str = "default",
        param: QueryParam = QueryParam(),
        filter: Optional[Dict] = None
    ) -> Optional[QueryResult]:
        """
        Asynchronously perform string-based query on the specified collection using LightRAG.

        Args:
            query (str, optional): The query string. Defaults to an empty string.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            param (QueryParam, optional): Query parameters. Defaults to QueryParam().
            filter (Optional[Dict], optional): Filter criteria for the query. Defaults to None.

        Returns:
            Optional[QueryResult]: The search result containing document IDs, embeddings, metadatas, documents, distances, or an error message encapsulated in QueryResult.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting query_async with query: '{query}', collection_name: '{collection_name}', param: {param}, filter: {filter}")

        try:
            lightrag_instance = self.client.get_lightRAG_instance(collection_name)

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
            logger.exception(f"[Request ID: {request_id}] Exception during query_async: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=str(e))

    async def vector_search_async(
        self,
        vectors: List[Union[List[float], list]],
        *,
        collection_name: str = "default",
        limit: Optional[int] = 5
    ) -> Optional[QueryResult]:
        """
        Asynchronously perform vector search on the specified collection.

        Args:
            vectors (List[Union[List[float], list]]): List of query vectors.
            collection_name (str, optional): The name of the collection. Defaults to "default".
            limit (Optional[int], optional): Maximum number of results to return per vector. Defaults to 5.

        Returns:
            Optional[QueryResult]: The search result.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"[Request ID: {request_id}] Starting vector_search_async with vectors: {vectors}, collection_name: '{collection_name}', limit: {limit}")

        try:
            lightrag_instance = self.client.get_lightRAG_instance(collection_name)

            # Validate vectors
            if not isinstance(vectors, list) or not all(isinstance(vec, (list, tuple)) for vec in vectors):
                error_msg = "Vectors must be a list of lists or tuples of floats."
                logger.error(f"[Request ID: {request_id}] {error_msg} Received type: {type(vectors)} with elements types: {[type(vec) for vec in vectors]}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            if not vectors:
                error_msg = "Vectors list cannot be empty."
                logger.warning(f"[Request ID: {request_id}] {error_msg}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            logger.debug(f"[Request ID: {request_id}] Performing vector search with limit: {limit}")
            response = await lightrag_instance.avector_search(vectors, limit=limit)
            logger.debug(f"[Request ID: {request_id}] Received vector search response: {response} (type: {type(response)})")

            # Check if response has 'ids' attribute
            if hasattr(response, 'ids') and response.ids and len(response.ids) > 0:
                logger.debug(f"[Request ID: {request_id}] Returning QueryResult with IDs: {response.ids}")
                # Calculate distances if necessary; for now, use placeholders
                distances = [[0.0 for _ in ids] for ids in response.ids]  # Placeholder distances

                # Fetch documents and metadata for each list of IDs
                documents = []
                metadatas = []
                for id_list in response.ids:
                    docs = await lightrag_instance.afetch_documents(id_list)
                    metas = await lightrag_instance.afetch_metadatas(id_list)
                    documents.append(docs)
                    metadatas.append(metas)

                return QueryResult(
                    ids=response.ids,
                    embeddings=response.embeddings,
                    metadatas=metadatas,
                    documents=documents,
                    distances=distances,  # Include distances
                    error=None
                )
            else:
                error_msg = "No documents found for the given vector search."
                logger.error(f"[Request ID: {request_id}] {error_msg}")
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
        except Exception as e:
            logger.exception(f"[Request ID: {request_id}] Exception during vector_search_async: {e}")
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=str(e))

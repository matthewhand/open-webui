# search.py

import uuid
import json
from typing import Optional, List, Dict, Any, Union
from logging import getLogger, DEBUG

from lightrag import LightRAG, QueryParam
from .utils import QueryResult

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
        logger.debug("SearchHandler initialized with LightRAGClient.", extra={'request_id': 'N/A'})

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
        logger.debug(f"Starting query_async with query: '{query}', collection_name: '{collection_name}', param: {param}, filter: {filter}", extra={'request_id': request_id})

        try:
            lightrag_instance = self.client.get_lightRAG_instance(collection_name)

            # Validate query
            if not isinstance(query, str):
                error_msg = "Query must be a string."
                logger.error(f"{error_msg} Received type: {type(query)}", extra={'request_id': request_id})
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            if not query.strip():
                error_msg = "Query string cannot be empty."
                logger.warning(f"{error_msg}", extra={'request_id': request_id})
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            if filter:
                # Validate filter structure
                if not isinstance(filter, dict):
                    error_msg = "Filter must be a dictionary."
                    logger.error(f"{error_msg} Received type: {type(filter)}", extra={'request_id': request_id})
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
                if 'excluded_ids' in filter and not isinstance(filter['excluded_ids'], list):
                    error_msg = "'excluded_ids' in filter must be a list."
                    logger.error(f"{error_msg} Received type: {type(filter['excluded_ids'])}", extra={'request_id': request_id})
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

                logger.debug(f"Performing query with filter: {filter}", extra={'request_id': request_id})
                response = await lightrag_instance.aquery(query, param)
                logger.debug(f"Received response from aquery with filter: {response} (type: {type(response)})", extra={'request_id': request_id})

                # Check if response has 'ids' attribute
                if hasattr(response, 'ids') and response.ids and len(response.ids) > 0:
                    excluded_ids = filter.get('excluded_ids', [])
                    logger.debug(f"Excluded IDs: {excluded_ids}", extra={'request_id': request_id})

                    # Filter out excluded IDs
                    filtered_ids = [doc_id for doc_id in response.ids[0] if doc_id not in excluded_ids]
                    logger.debug(f"Filtered IDs count: {len(filtered_ids)} out of {len(response.ids[0])}", extra={'request_id': request_id})

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
                    logger.error(f"{error_msg}", extra={'request_id': request_id})
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
            else:
                # Perform a standard generative query
                logger.debug("Performing standard query without filter.", extra={'request_id': request_id})
                response = await lightrag_instance.aquery(query, param)
                logger.debug(f"Received query response: {response} (type: {type(response)})", extra={'request_id': request_id})

                # Check if response has 'ids' attribute
                if hasattr(response, 'ids') and response.ids and len(response.ids) > 0:
                    logger.debug(f"Returning QueryResult with IDs: {response.ids}", extra={'request_id': request_id})
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
                            logger.error(f"{error_msg} Content: {response}", extra={'request_id': request_id})
                            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
                    except json.JSONDecodeError:
                        error_msg = "Unexpected string response from query."
                        logger.error(f"{error_msg} Content: {response}", extra={'request_id': request_id})
                        return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
                else:
                    error_msg = "Unexpected response format from query."
                    logger.error(f"{error_msg} Response type: {type(response)}. Content: {response}", extra={'request_id': request_id})
                    return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)
        except Exception as e:
            logger.error(f"Query failed for '{query}' with error: {e}", extra={'request_id': request_id})
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Internal server error during query processing.")
        finally:
            logger.debug("Completed query_async.", extra={'request_id': request_id})
            await self._query_done(collection_name)

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
        truncated_vectors = truncate_vectors_for_logging(vectors)
        logger.debug(f"Starting vector_search_async with vectors: {truncated_vectors}, limit: {limit}, collection_name: '{collection_name}'", extra={'request_id': request_id})

        try:
            lightrag_instance = self.client.get_lightRAG_instance(collection_name)

            # Access the appropriate vector storage; e.g., 'chunks_vdb'
            vector_storage = getattr(lightrag_instance, 'chunks_vdb', None)
            if vector_storage is None:
                error_msg = f"Vector storage 'chunks_vdb' not found in collection '{collection_name}'."
                logger.error(f"{error_msg}", extra={'request_id': request_id})
                return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error=error_msg)

            all_doc_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []
            all_distances = []

            for idx, vector in enumerate(vectors):
                truncated_vector = truncate_vector(vector)
                logger.debug(f"Processing vector {idx + 1}/{len(vectors)}: {truncated_vector}", extra={'request_id': request_id})
                try:
                    # Ensure the vector is in list format and has correct dimensions
                    if isinstance(vector, (list, tuple)):
                        query_vector = vector
                        logger.debug(f"Vector {idx + 1} is already a list.", extra={'request_id': request_id})
                    else:
                        logger.error(f"Unsupported vector type: {type(vector)}. Skipping this vector.", extra={'request_id': request_id})
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])
                        continue

                    # Validate embedding dimensions
                    if len(query_vector) != self.embedding_func.embedding_dim:
                        error_msg = f"Vector {idx + 1} has incorrect dimensions: {len(query_vector)}. Expected: {self.embedding_func.embedding_dim}."
                        logger.error(f"{error_msg}", extra={'request_id': request_id})
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])
                        continue

                    # Log the query vector for debugging (truncated)
                    logger.debug(f"Query Vector {idx + 1}: {truncated_vector}", extra={'request_id': request_id})

                    # Perform similarity search using the vector storage's query method
                    search_results = await vector_storage._client.query(
                        query=query_vector,
                        top_k=limit,
                        better_than_threshold=0.0  # Temporarily set to 0.0 to include all results
                    )
                    logger.debug(f"Retrieved {len(search_results)} results for vector {idx + 1}", extra={'request_id': request_id})

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

                    logger.debug(f"Aggregated results for vector {idx + 1}: {doc_ids}", extra={'request_id': request_id})
                except Exception as e:
                    logger.error(f"Error during vector search for vector {idx + 1}: {e}", extra={'request_id': request_id})
                    all_doc_ids.append([])
                    all_embeddings.append([])
                    all_metadatas.append([])
                    all_documents.append([])
                    all_distances.append([])

            logger.debug("Vector search completed for all vectors.", extra={'request_id': request_id})
            return QueryResult(
                ids=all_doc_ids,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                documents=all_documents,
                distances=all_distances,  # Include distances
                error=None
            )
        except Exception as e:
            logger.error(f"Vector search failed with error: {e}", extra={'request_id': request_id})
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Internal server error during vector search.")
        finally:
            logger.debug("Completed vector_search_async.", extra={'request_id': request_id})
            await self._query_done(collection_name)

    async def _query_done(self, collection_name: str):
        """
        Handle post-query operations.

        Args:
            collection_name (str): The name of the collection.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Starting _query_done for collection '{collection_name}'", extra={'request_id': request_id})
        try:
            lightrag_instance = self.client.get_lightRAG_instance(collection_name)
            await lightrag_instance._query_done()
            logger.debug(f"Completed query operations for collection '{collection_name}'.", extra={'request_id': request_id})
        except Exception as e:
            logger.error(f"Error in _query_done for collection '{collection_name}': {e}", extra={'request_id': request_id})

    # Additional search-related methods can be added here as needed.

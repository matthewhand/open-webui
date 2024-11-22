# lightrag_client.py

import uuid
from typing import Optional, List, Dict, Any, Union
from .custom_lightrag import CustomLightRAG
from .utils import QueryResult
from logging import getLogger, DEBUG

logger = getLogger("lightrag_client")
logger.setLevel(DEBUG)

class CustomLightRAGClient:
    def __init__(self):
        self.lightrag_instance = CustomLightRAG()
        # Initialize any other necessary components

    async def vector_search_async(
        self,
        vectors: List[Union[List[float], list]],
        *,
        collection_name: str = "default",
        limit: Optional[int] = 5
    ) -> Optional[QueryResult]:
        request_id = str(uuid.uuid4())
        logger.debug(f"Starting vector_search_async with vectors: {vectors[:2]}..., limit: {limit}, collection_name: '{collection_name}'", extra={'request_id': request_id})

        try:
            lightrag_instance = self.lightrag_instance
            vector_storage: CustomNanoVectorDBStorage = lightrag_instance.chunks_vdb

            all_doc_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []
            all_distances = []

            for idx, vector in enumerate(vectors):
                logger.debug(f"Processing vector {idx + 1}/{len(vectors)}: {vector[:3]}...", extra={'request_id': request_id})
                try:
                    if not isinstance(vector, (list, tuple)):
                        logger.error(f"Unsupported vector type: {type(vector)}. Skipping this vector.", extra={'request_id': request_id})
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])
                        continue

                    if len(vector) != lightrag_instance.embedding_func.embedding_dim:
                        logger.error(f"Vector {idx + 1} has incorrect dimensions: {len(vector)}. Expected: {lightrag_instance.embedding_func.embedding_dim}.", extra={'request_id': request_id})
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])
                        continue

                    logger.debug(f"Query Vector {idx + 1}: {vector[:3]}...", extra={'request_id': request_id})  # Truncated for logging

                    # Perform vector-based search using the custom vector_query method
                    search_results = await vector_storage.vector_query(vector, top_k=limit)
                    logger.debug(f"Retrieved {len(search_results)} results for vector {idx + 1}", extra={'request_id': request_id})
                    logger.debug(f"Search results structure for vector {idx + 1}: {search_results}", extra={'request_id': request_id})

                    if search_results:
                        doc_ids = [result.get('id') for result in search_results]
                        distances = [result.get('distance') for result in search_results]

                        # Fetch embeddings, metadata, and documents
                        embeddings = await self.get_embeddings(doc_ids)
                        metadatas = await self.get_metadatas(doc_ids)
                        documents = await self.get_documents(doc_ids)

                        all_doc_ids.append(doc_ids)
                        all_embeddings.append(embeddings)
                        all_metadatas.append(metadatas)
                        all_documents.append(documents)
                        all_distances.append(distances)
                    else:
                        logger.warning(f"No results found for vector {idx + 1}", extra={'request_id': request_id})
                        all_doc_ids.append([])
                        all_embeddings.append([])
                        all_metadatas.append([])
                        all_documents.append([])
                        all_distances.append([])

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
                distances=all_distances,
                error=None
            )

        except Exception as e:
            logger.error(f"Vector search failed with error: {e}", extra={'request_id': request_id})
            return QueryResult(ids=[[]], embeddings=None, metadatas=None, documents=None, distances=[[]], error="Internal server error during vector search.")
        finally:
            logger.debug("Completed vector_search_async.", extra={'request_id': request_id})
            # Perform any necessary cleanup or callbacks

    def vector_search(
        self,
        vectors: List[Union[List[float], list]],
        *,
        collection_name: str = "default",
        limit: Optional[int] = 5
    ) -> Optional[QueryResult]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.vector_search_async(vectors, collection_name=collection_name, limit=limit))

    # Implement helper methods to fetch embeddings, metadatas, and documents
    async def get_embeddings(self, doc_ids: List[str]) -> List[List[float]]:
        return await self.lightrag_instance.chunks_vdb.get_embeddings(doc_ids)

    async def get_metadatas(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        return await self.lightrag_instance.chunks_vdb.get_metadatas(doc_ids)

    async def get_documents(self, doc_ids: List[str]) -> List[str]:
        return await self.lightrag_instance.chunks_vdb.get_documents(doc_ids)

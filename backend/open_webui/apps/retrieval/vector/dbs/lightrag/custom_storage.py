# custom_storage.py

import asyncio
from typing import List, Dict, Any, Optional
from lightrag.storage import NanoVectorDBStorage  # Adjust the import path as necessary
from lightrag.utils import logger
import numpy as np  # Ensure numpy is imported if using np.ndarray


class CustomNanoVectorDBStorage(NanoVectorDBStorage):
    async def vector_query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a vector-based query and map the results to the expected structure.

        Args:
            vector (List[float]): The query vector.
            top_k (int): Number of top results to retrieve.

        Returns:
            List[Dict[str, Any]]: List of search result dictionaries with mapped keys.
        """
        truncated_vector = vector[:3]  # For logging brevity
        logger.debug(f"Performing custom vector-based query with vector: {truncated_vector}...")

        try:
            results = await super().query(vector, top_k=top_k)
        except Exception as e:
            logger.error(f"Error during vector_query: {e}")
            raise

        # Map '__id__' to 'id' and '__metrics__' to 'distance'
        mapped_results = []
        for result in results:
            if not isinstance(result, dict):
                logger.warning(f"Unexpected result format: {result}")
                continue
            mapped_result = {
                "id": result.get("__id__", "unknown_id"),
                "distance": result.get("__metrics__", 0.0),
                **{k: v for k, v in result.items() if k not in ["__id__", "__metrics__"]}
            }
            mapped_results.append(mapped_result)

        logger.debug(f"Custom vector-based query results: {mapped_results}")
        return mapped_results

    async def get_embeddings(self, doc_ids: List[str]) -> List[List[float]]:
        """
        Fetch embeddings for the given document IDs.

        Args:
            doc_ids (List[str]): List of document IDs.

        Returns:
            List[List[float]]: List of embeddings corresponding to the document IDs.
        """
        logger.debug(f"Fetching embeddings for doc_ids: {doc_ids}")
        embeddings = await self._fetch_data(doc_ids, key="__vector__")
        processed_embeddings = self._process_embeddings(embeddings)
        return processed_embeddings

    async def get_metadatas(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch metadata for the given document IDs.

        Args:
            doc_ids (List[str]): List of document IDs.

        Returns:
            List[Dict[str, Any]]: List of metadata dictionaries corresponding to the document IDs.
        """
        logger.debug(f"Fetching metadatas for doc_ids: {doc_ids}")
        metadatas = await self._fetch_data(doc_ids, key="metadata")
        processed_metadatas = self._process_metadatas(metadatas)
        return processed_metadatas

    async def get_documents(self, doc_ids: List[str]) -> List[str]:
        """
        Fetch document texts for the given document IDs.

        Args:
            doc_ids (List[str]): List of document IDs.

        Returns:
            List[str]: List of document contents corresponding to the document IDs.
        """
        logger.debug(f"Fetching documents for doc_ids: {doc_ids}")
        documents = await self._fetch_data(doc_ids, key="content")
        processed_documents = self._process_documents(documents)
        return processed_documents

    async def _fetch_data(self, doc_ids: List[str], key: str) -> List[Optional[Any]]:
        """
        Helper method to fetch data for a given key from multiple document IDs concurrently.

        Args:
            doc_ids (List[str]): List of document IDs.
            key (str): The key to extract from each document.

        Returns:
            List[Optional[Any]]: List of extracted values corresponding to the key.
        """
        logger.debug(f"Fetching '{key}' for doc_ids: {doc_ids}")
        tasks = [self._client.get(doc_id) for doc_id in doc_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fetched_data = []
        for doc_id, result in zip(doc_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching '{key}' for doc_id {doc_id}: {result}")
                fetched_data.append(None)
                continue
            if result and key in result:
                fetched_data.append(result[key])
            else:
                fetched_data.append(None)
                logger.warning(f"'{key}' not found for doc_id: {doc_id}")
        logger.debug(f"Fetched '{key}': {fetched_data}")
        return fetched_data

    def _process_embeddings(self, embeddings: List[Optional[Any]]) -> List[List[float]]:
        """
        Process raw embeddings into standardized lists of floats.

        Args:
            embeddings (List[Optional[Any]]): Raw embeddings data.

        Returns:
            List[List[float]]: Processed embeddings.
        """
        processed = []
        for emb in embeddings:
            if isinstance(emb, list):
                processed.append(emb)
            elif isinstance(emb, tuple):
                processed.append(list(emb))
            elif isinstance(emb, np.ndarray):
                processed.append(emb.tolist())
            else:
                processed.append([])
                logger.warning(f"Embedding of unsupported type: {type(emb)}. Appended empty list.")
        logger.debug(f"Processed embeddings: {processed}")
        return processed

    def _process_metadatas(self, metadatas: List[Optional[Any]]) -> List[Dict[str, Any]]:
        """
        Process raw metadatas into standardized dictionaries.

        Args:
            metadatas (List[Optional[Any]]): Raw metadatas data.

        Returns:
            List[Dict[str, Any]]: Processed metadatas.
        """
        processed = []
        for meta in metadatas:
            if isinstance(meta, dict):
                # Exclude specific keys if necessary
                processed_meta = {k: v for k, v in meta.items() if k not in ["__id__", "__vector__", "content"]}
                processed.append(processed_meta)
            else:
                processed.append({})
                logger.warning(f"Metadata is not a dict or is None. Appended empty dict.")
        logger.debug(f"Processed metadatas: {processed}")
        return processed

    def _process_documents(self, documents: List[Optional[Any]]) -> List[str]:
        """
        Process raw documents into standardized strings.

        Args:
            documents (List[Optional[Any]]): Raw documents data.

        Returns:
            List[str]: Processed documents.
        """
        processed = []
        for doc in documents:
            if isinstance(doc, str):
                processed.append(doc)
            else:
                processed.append("")
                logger.warning(f"Document content is not a string or is None. Appended empty string.")
        logger.debug(f"Processed documents: {processed}")
        return processed_documents

# lightrag_search.py

import uuid
from typing import Optional, List, Dict, Any, Union
from logging import getLogger, DEBUG

from lightrag import LightRAG
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
    ```

### **Explanation of Changes:**

1. **Maintained Separate `query_async` and `query` Methods:**
   - **`query_async`:** Handles string-based queries with optional `filter`.
   - **`query`:** Synchronously calls `query_async`.

2. **Ensured `query_async` Does Not Require `vectors`:**
   - The `query_async` method focuses solely on string-based queries, ensuring that it doesn't expect `vectors` as an argument.

3. **Maintained `filter` Functionality:**
   - The `filter` parameter is optional and integrated into the `query_async` method to refine search results.

4. **Removed `filter` from `query_async` in Previous Responses:**
   - Ensured that `filter` remains within `query_async` without affecting the `vector_search` methods.

---

## **3. Review and Update All Calls to `query` and `vector_search`**

Now that the `query` and `vector_search` methods are separated, ensure that all parts of your codebase use them appropriately.

### **Example Update in `main.py`**

Assuming you have a function `save_docs_to_vector_db` that interacts with the vector DB, update it as follows:

```python
# main.py

from open_webui.apps.retrieval.vector.dbs.lightrag import LightRAGClient
from open_webui.apps.retrieval.vector.dbs.lightrag_utils import QueryResult
from fastapi import HTTPException

# Initialize the client
VECTOR_DB_CLIENT = LightRAGClient()

def save_docs_to_vector_db(docs: List[str], collection_name: str, filter_criteria: Optional[Dict] = None) -> QueryResult:
    """
    Save documents to the vector DB and perform a query with optional filtering.

    Args:
        docs (List[str]): List of document texts to insert.
        collection_name (str): Name of the collection to insert documents into.
        filter_criteria (Optional[Dict], optional): Filter criteria for querying. Defaults to None.

    Returns:
        QueryResult: Result of the query.
    """
    try:
        # Insert documents
        insert_success = VECTOR_DB_CLIENT.insert(items=docs, collection_name=collection_name)
        if not insert_success:
            raise ValueError("Failed to insert documents into the vector DB.")

        # Compute embeddings
        embeddings = VECTOR_DB_CLIENT.embedding_func.func(docs)

        # Perform vector search with filters if needed
        if filter_criteria:
            query_result = VECTOR_DB_CLIENT.query(
                query="",  # Empty query string since we're performing a vector search
                collection_name=collection_name,
                filter=filter_criteria
            )
        else:
            # If no filter is needed, you might perform a standard query or vector search
            # Depending on your application logic
            query_result = VECTOR_DB_CLIENT.query(
                query="",
                collection_name=collection_name
            )

        return query_result
    except Exception as e:
        logger.error(f"Error in save_docs_to_vector_db: {e}")
        raise e

# Example usage within an API endpoint
async def process_file(form: ProcessFileForm):
    try:
        docs = extract_docs(form.file_id)  # Implement this function as needed
        filter_criteria = {"category": {"$eq": "science"}}  # Example filter
        result = save_docs_to_vector_db(docs, "default_collection", filter_criteria)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

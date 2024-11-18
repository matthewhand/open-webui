# utils.py

from typing import Optional, Dict, Any, Union, Callable, Awaitable, List
from pydantic import BaseModel, Field
import json
import numpy as np
import logging

logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG)

class QueryResult(BaseModel):
    """
    Data model for encapsulating the results of a query or search operation.
    """
    ids: List[List[str]] = Field(
        default_factory=list,
        description="List of document IDs for each query vector."
    )
    embeddings: Optional[List[List[float]]] = Field(
        default=None,
        description="List of embeddings for each document."
    )
    metadatas: Optional[List[List[Dict[str, Any]]]] = Field(
        default=None,
        description="List of metadata dictionaries for each document."
    )
    documents: Optional[List[List[str]]] = Field(
        default=None,
        description="List of document texts."
    )
    distances: Optional[List[List[float]]] = Field(
        default_factory=list,
        description="List of distances/similarities for each document."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message, if any."
    )

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types if needed

    def dict(self, **kwargs):
        """
        Override dict() to handle serialization of embeddings and distances.
        """
        data = super().dict(**kwargs)
        if self.embeddings is not None:
            data['embeddings'] = [embedding for embedding in self.embeddings]
        if self.distances is not None:
            data['distances'] = [distance for distance in self.distances]
        return data

class EmbeddingFunc(BaseModel):
    """
    Data model for embedding functions, specifying embedding dimensions and the function to generate embeddings.
    """
    embedding_dim: int
    max_token_size: int
    func: Callable[..., Awaitable[List[List[float]]]]  # Specify that it's an async callable returning embeddings

    async def __call__(self, *args, **kwargs) -> List[List[float]]:
        # Call the wrapped function
        embeddings = await self.func(*args, **kwargs)
        logger.debug(f"Embeddings received in EmbeddingFunc: {type(embeddings)}")

        # Ensure embeddings are lists
        if isinstance(embeddings, np.ndarray):
            logger.warning("Embeddings are NumPy arrays. Converting to lists.")
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, list):
            # Convert any inner numpy arrays to lists
            embeddings = [
                vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in embeddings
            ]
            logger.debug("Ensured all inner embeddings are lists.")
        else:
            logger.error(f"Unexpected embedding type: {type(embeddings)}")
            raise TypeError(f"Embeddings must be a list or NumPy array, got {type(embeddings)}")

        # Final type check
        if not all(isinstance(vec, list) for vec in embeddings):
            logger.error("One or more embeddings are not lists after conversion.")
            raise TypeError("All embeddings must be lists of floats.")

        logger.debug("Embeddings successfully converted to lists within EmbeddingFunc.")
        return embeddings

def convert_response_to_json(response: Any) -> Dict[str, Any]:
    """
    Convert a response object to a JSON-serializable dictionary.

    Args:
        response (Any): The response object to convert.

    Returns:
        Dict[str, Any]: JSON-serializable dictionary representation of the response.
    """
    try:
        if hasattr(response, 'dict'):
            return response.dict()
        elif isinstance(response, str):
            return json.loads(response)
        else:
            # For other types, attempt to convert using repr
            return {"data": repr(response)}
    except Exception as e:
        # Handle any exceptions during conversion
        return {"error": f"Failed to convert response to JSON: {e}"}

def truncate_vector(vector: Union[List[float], list], max_elements: int = 3) -> str:
    """
    Truncate a vector to the first 'max_elements' elements for logging purposes.

    Args:
        vector (Union[List[float], list]): The vector to truncate.
        max_elements (int, optional): Number of elements to keep. Defaults to 3.

    Returns:
        str: Truncated vector as a string.
    """
    if not isinstance(vector, (list, tuple)):
        return "Invalid vector type"
    truncated = vector[:max_elements]
    return ', '.join(map(str, truncated)) + ('...' if len(vector) > max_elements else '')

def truncate_vectors_for_logging(vectors: Optional[List[Union[List[float], list]]], max_elements: int = 3) -> Optional[List[str]]:
    """
    Truncate all vectors in a list for logging purposes.

    Args:
        vectors (Optional[List[Union[List[float], list]]]): List of vectors to truncate.
        max_elements (int, optional): Number of elements to keep in each vector. Defaults to 3.

    Returns:
        Optional[List[str]]: List of truncated vectors as strings.
    """
    if vectors is None:
        return None
    return [truncate_vector(vector, max_elements) for vector in vectors]

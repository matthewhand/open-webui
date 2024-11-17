# lightrag_utils.py

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json
from typing import Union, Callable

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

    model_config = {
        "arbitrary_types_allowed": True  # Allow arbitrary types if needed
    }

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
    func: Callable  # Using Callable for better type hinting

    model_config = {
        "arbitrary_types_allowed": True  # Allow callable types
    }

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

# llm.py

import os
import asyncio
import numpy as np
from typing import List
import logging

# Ensure the logger is properly configured
logger = logging.getLogger("wrapped_openai_embedding")
logger.setLevel(logging.DEBUG)

# Original embedding function (assumed to be provided)
# from lightrag.llm import openai_embedding

async def openai_embedding(documents: List[str]) -> List[List[float]]:
    """
    Asynchronously generate embeddings for a list of documents using OpenAI's embedding API.

    Args:
        documents (List[str]): The list of documents to embed.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    # Placeholder for the actual OpenAI embedding call
    # Replace with actual implementation
    # Example:
    # response = await openai.Embedding.acreate(input=documents, model="text-embedding-ada-002")
    # embeddings = [item['embedding'] for item in response['data']]
    # return embeddings

    # For demonstration purposes, return dummy embeddings of correct dimension
    embedding_dim = 1536
    return [[0.1] * embedding_dim for _ in documents]

async def wrapped_openai_embedding(documents: List[str]) -> List[List[float]]:
    """
    Wrapper around the original openai_embedding function to ensure embeddings are lists.

    Args:
        documents (List[str]): The list of documents to embed.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    # Call the original embedding function
    embeddings = await openai_embedding(documents)

    # Debug log the type of embeddings received
    logger.debug(f"Original embeddings type: {type(embeddings)}")

    # Check and convert if necessary
    if isinstance(embeddings, np.ndarray):
        logger.debug("Converting embeddings from NumPy array to list.")
        embeddings = embeddings.tolist()
    elif isinstance(embeddings, list):
        # Further ensure that inner elements are lists of floats
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

    logger.debug("Embeddings successfully converted to lists.")
    return embeddings

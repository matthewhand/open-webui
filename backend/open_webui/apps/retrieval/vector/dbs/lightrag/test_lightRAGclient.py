# test_lightRAGclient.py

import pytest
from .client import LightRAGClient
from .utils import QueryResult

@pytest.fixture
def client():
    return LightRAGClient()

@pytest.mark.asyncio
async def test_insert_single_document_async(client):
    document = "This is a test document."
    result = await client.insert_async(items=document, collection_name="test_collection")
    assert result == True

@pytest.mark.asyncio
async def test_insert_multiple_documents_async(client):
    documents = ["Doc 1", "Doc 2", "Doc 3"]
    result = await client.insert_async(items=documents, collection_name="test_collection")
    assert result == True

@pytest.mark.asyncio
async def test_insert_documents_from_dict_async(client):
    documents_dict = {
        "doc1": "Document 1",
        "doc2": "Document 2",
        "doc3": "Document 3"
    }
    result = await client.insert_async(items=documents_dict, collection_name="test_collection")
    assert result == True

@pytest.mark.asyncio
async def test_query_without_filter_async(client):
    query_result = await client.query_async(
        query="test",
        collection_name="test_collection"
    )
    assert isinstance(query_result, QueryResult)
    assert query_result.error is None
    assert len(query_result.ids[0]) > 0

@pytest.mark.asyncio
async def test_query_with_filter_async(client):
    filter_criteria = {"excluded_ids": ["doc1"]}
    query_result = await client.query_async(
        query="AI",
        collection_name="test_collection",
        filter=filter_criteria
    )
    assert isinstance(query_result, QueryResult)
    assert query_result.error is None
    # Further assertions based on expected filtered results
    for doc_id in query_result.ids[0]:
        assert doc_id not in filter_criteria["excluded_ids"]

@pytest.mark.asyncio
async def test_vector_search_async(client):
    documents = ["AI in healthcare.", "Machine learning applications.", "Deep learning models."]
    insert_success = await client.insert_async(items=documents, collection_name="test_collection")
    assert insert_success == True

    embeddings = await client.embedding_func(documents)
    vector_search_result = await client.vector_search_async(
        vectors=embeddings,
        collection_name="test_collection",
        limit=2
    )
    assert isinstance(vector_search_result, QueryResult)
    assert vector_search_result.error is None
    assert len(vector_search_result.ids[0]) == 2

@pytest.mark.asyncio
async def test_delete_entity_async(client):
    entity_name = "TestEntity"
    result = await client.delete_by_entity_async(entity_name, collection_name="test_collection")
    assert result == True

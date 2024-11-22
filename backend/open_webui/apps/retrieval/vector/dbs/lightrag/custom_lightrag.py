# custom_lightrag.py

import os
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Union, List, Dict, Any, Optional, Callable

from lightrag.llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    local_query,
    global_query,
    hybrid_query,
    naive_query,
)
from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
    QueryResult,  # Importing QueryResult
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from lightrag.storage import (
    JsonKVStorage,
    NetworkXStorage,
)
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

# Import the custom storage class
from lightrag.custom_storage import CustomNanoVectorDBStorage


@dataclass
class CustomLightRAG:
    """
    CustomLightRAG extends the functionality of LightRAG by integrating custom storage mechanisms and handling 
    insertion, querying, and deletion of documents and entities.
    """
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="CustomNanoVectorDBStorage")  # Use custom storage
    graph_storage: str = field(default="NetworkXStorage")

    # Logging
    log_level: str = field(default="DEBUG")

    # Embedding and LLM configurations
    embedding_func_max_async: int = 16
    llm_model_max_async: int = 16
    llm_model_kwargs: Dict = field(default_factory=dict)
    enable_llm_cache: bool = True

    # Chunking configurations
    chunk_overlap_token_size: int = 50  # Define a reasonable default
    chunk_token_size: int = 500  # Define a reasonable default
    tiktoken_model_name: str = "gpt-3"  # Define the tokenizer/model name

    # Storage Classes (Initialized in __post_init__)
    key_string_value_json_storage_cls: Type[BaseKVStorage] = field(init=False)
    vector_db_storage_cls: Type[BaseVectorStorage] = field(init=False)
    graph_storage_cls: Type[BaseGraphStorage] = field(init=False)

    # Actual Storage Instances
    llm_response_cache: Optional[BaseKVStorage] = field(init=False, default=None)
    full_docs: BaseKVStorage = field(init=False)
    text_chunks: BaseKVStorage = field(init=False)
    chunk_entity_relation_graph: BaseGraphStorage = field(init=False)

    entities_vdb: BaseVectorStorage = field(init=False)
    relationships_vdb: BaseVectorStorage = field(init=False)
    chunks_vdb: BaseVectorStorage = field(init=False)

    llm_model_func: Callable = field(init=False)

    def __post_init__(self):
        """
        Initialize the CustomLightRAG instance by setting up logging, storage classes, and storage instances.
        """
        # Setup logging
        log_file = os.path.join(self.working_dir, "lightrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: '{self.working_dir}'")
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items() if not k.startswith("_")])
        logger.debug(f"CustomLightRAG initialized with parameters:\n  {_print_config}\n")

        # Initialize storage classes
        storage_classes = self._get_storage_classes()

        self.key_string_value_json_storage_cls = storage_classes[self.kv_storage]
        self.vector_db_storage_cls = storage_classes[self.vector_storage]
        self.graph_storage_cls = storage_classes[self.graph_storage]

        # Ensure working directory exists
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory '{self.working_dir}'")
            os.makedirs(self.working_dir, exist_ok=True)

        # Initialize storage instances
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )

        # Wrap embedding function to limit concurrency
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            openai_embedding  # Ensure this is the correct embedding function
        )

        # Initialize full_docs and text_chunks storage
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # Initialize vector storages using custom storage class
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # Initialize LLM model function with async limits and additional kwargs
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                gpt_4o_mini_complete,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_classes(self) -> Dict[str, Type[Union[BaseKVStorage, BaseVectorStorage, BaseGraphStorage]]]:
        """
        Retrieve the storage classes based on storage type names.

        Returns:
            Dict[str, Type]: Mapping of storage type names to their corresponding classes.
        """
        return {
            # KV storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            # Vector storage
            "CustomNanoVectorDBStorage": CustomNanoVectorDBStorage,  # Custom storage
            "OracleVectorDBStorage": OracleVectorDBStorage,
            # Graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage  # Uncomment if needed
        }

    async def ainsert(self, string_or_strings: Union[str, List[str]]):
        """
        Asynchronously insert documents into storage.

        Args:
            string_or_strings (Union[str, List[str]]): Single string or list of strings to insert.
        """
        update_storage = False
        try:
            # Normalize input to list of strings
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            logger.debug(f"Prepared new documents: {new_docs}", extra={'request_id': 'N/A'})

            # Filter out already existing documents
            existing_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k not in existing_keys}
            if not new_docs:
                logger.warning("All documents are already in the storage.", extra={'request_id': 'N/A'})
                return
            update_storage = True
            logger.info(f"Inserting {len(new_docs)} new documents.", extra={'request_id': 'N/A'})

            # Chunk documents
            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(chunk["content"], prefix="chunk-"): {
                        **chunk,
                        "full_doc_id": doc_key,
                    }
                    for chunk in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            logger.debug(f"Prepared chunks: {inserting_chunks}", extra={'request_id': 'N/A'})

            # Filter out already existing chunks
            existing_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k not in existing_chunk_keys}
            if not inserting_chunks:
                logger.warning("All chunks are already in the storage.", extra={'request_id': 'N/A'})
                return
            logger.info(f"Inserting {len(inserting_chunks)} new chunks.", extra={'request_id': 'N/A'})

            # Upsert chunks into vector storage
            await self.chunks_vdb.upsert(inserting_chunks)
            logger.debug("Chunks upserted successfully.", extra={'request_id': 'N/A'})

            # Extract entities and relationships
            logger.info("Extracting entities and relationships...", extra={'request_id': 'N/A'})
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                text_chunks=self.text_chunks,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found.", extra={'request_id': 'N/A'})
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            logger.debug("Entities and relationships extracted successfully.", extra={'request_id': 'N/A'})

            # Upsert documents and chunks
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            logger.info("Documents and chunks upserted successfully.", extra={'request_id': 'N/A'})
        except Exception as e:
            logger.error(f"Error during insertion: {e}", extra={'request_id': 'N/A'})
            raise
        finally:
            if update_storage:
                await self._insert_done()

    def insert(self, string_or_strings: Union[str, List[str]]):
        """
        Synchronously insert documents by running the asynchronous ainsert method.

        Args:
            string_or_strings (Union[str, List[str]]): Single string or list of strings to insert.
        """
        asyncio.run(self.ainsert(string_or_strings))

    async def _insert_done(self):
        """
        Handle post-insertion operations by calling index_done_callback on all relevant storage instances.
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
        logger.debug("Post-insertion operations completed.", extra={'request_id': 'N/A'})

    async def aquery(self, query: str, param: QueryParam = QueryParam()) -> QueryResult:
        """
        Asynchronously perform a query based on the specified mode.

        Args:
            query (str): The query string.
            param (QueryParam): Query parameters including the mode.

        Returns:
            QueryResult: The result of the query.
        """
        try:
            if param.mode == "local":
                response = await local_query(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            elif param.mode == "global":
                response = await global_query(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            elif param.mode == "hybrid":
                response = await hybrid_query(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            elif param.mode == "naive":
                response = await naive_query(
                    query,
                    self.chunks_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            else:
                error_msg = f"Unknown mode '{param.mode}' provided for query."
                logger.error(error_msg, extra={'request_id': 'N/A'})
                return QueryResult(
                    ids=[[]],
                    embeddings=None,
                    metadatas=None,
                    documents=None,
                    distances=[[]],
                    error=error_msg
                )
            
            logger.debug(f"Query response: {response}", extra={'request_id': 'N/A'})
            await self._query_done()
            return response
        except Exception as e:
            logger.error(f"Error during query: {e}", extra={'request_id': 'N/A'})
            return QueryResult(
                ids=[[]],
                embeddings=None,
                metadatas=None,
                documents=None,
                distances=[[]],
                error=str(e)
            )

    def query(self, query: str, param: QueryParam = QueryParam()) -> QueryResult:
        """
        Synchronously perform a query by running the asynchronous aquery method.

        Args:
            query (str): The query string.
            param (QueryParam): Query parameters including the mode.

        Returns:
            QueryResult: The result of the query.
        """
        return asyncio.run(self.aquery(query, param))

    async def adelete_by_entity(self, entity_name: str):
        """
        Asynchronously delete an entity and its relationships.

        Args:
            entity_name (str): The name of the entity to delete.
        """
        formatted_entity = f'"{entity_name.upper()}"'
        try:
            await self.entities_vdb.delete_entity(formatted_entity)
            await self.relationships_vdb.delete_relation(formatted_entity)
            await self.chunk_entity_relation_graph.delete_node(formatted_entity)

            logger.info(f"Entity '{formatted_entity}' and its relationships have been deleted.", extra={'request_id': 'N/A'})
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{formatted_entity}': {e}", extra={'request_id': 'N/A'})

    def delete_by_entity(self, entity_name: str):
        """
        Synchronously delete an entity by running the asynchronous adelete_by_entity method.

        Args:
            entity_name (str): The name of the entity to delete.
        """
        asyncio.run(self.adelete_by_entity(entity_name))

    async def _delete_by_entity_done(self):
        """
        Handle post-deletion operations by calling index_done_callback on relevant storage instances.
        """
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
        logger.debug("Post-deletion operations completed.", extra={'request_id': 'N/A'})

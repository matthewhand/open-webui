from open_webui.config import VECTOR_DB

if VECTOR_DB == "milvus":
    from open_webui.apps.retrieval.vector.dbs.milvus import MilvusClient

    VECTOR_DB_CLIENT = MilvusClient()
elif VECTOR_DB == "qdrant":
    from open_webui.apps.retrieval.vector.dbs.qdrant import QdrantClient

    VECTOR_DB_CLIENT = QdrantClient()

elif VECTOR_DB == "lightrag":
    from open_webui.apps.retrieval.vector.dbs.lightrag import LightRAGClient
    VECTOR_DB_CLIENT = LightRAGClient()

else:
    from open_webui.apps.retrieval.vector.dbs.chroma import ChromaClient

    VECTOR_DB_CLIENT = ChromaClient()

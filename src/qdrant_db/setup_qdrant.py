from qdrant_client import QdrantClient
import os

qdrant_client = QdrantClient(
    url="https://4b6b6bd6-689d-4874-a9ab-f7f2489ee76b.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

print(qdrant_client.get_collections())
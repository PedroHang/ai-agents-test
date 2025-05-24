from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from strands import tool


QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = "https://4b6b6bd6-689d-4874-a9ab-f7f2489ee76b.us-east-1-0.aws.cloud.qdrant.io:6333" # User's Qdrant URL

def init_qdrant_client(qdrant_url: str, qdrant_api_key: str) -> QdrantClient:
    """
    Initializes a Qdrant client with the provided URL and API key.
    Args:
        qdrant_url (str): The URL of the Qdrant instance.
        qdrant_api_key (str): The API key for authentication.
    Returns:
        QdrantClient: An initialized Qdrant client.
    """     
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable not set.")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        print("Qdrant client initialized successfully.")
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Qdrant client: {e}")
    
@tool
def retrieve_relevant_texts(
    query: str,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    top_k: int = 5,
    score_threshold: float = None
) -> list:
    """
    Retrieves the most relevant text chunks from a Qdrant vector database
    based on the given text query.

    Args:
        query (str): The user query to search in the knowledge base.
        client (QdrantClient): An initialized Qdrant client.
        collection_name (str): The name of the Qdrant collection to search in.
        model (SentenceTransformer): The sentence-transformer model used for generating embeddings.
        top_k (int): The maximum number of relevant documents to retrieve.
        score_threshold (float, optional): If set, only results with a score
                                           equal to or above this threshold will be returned.
                                           Qdrant COSINE scores are between -1 and 1 (higher is better).
                                           For 'all-MiniLM-L6-v2', typical good scores are > 0.6 or 0.7.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - 'text': The retrieved text chunk.
            - 'source_pdf': The name of the PDF file the chunk came from.
            - 'chunk_number': The number of the chunk within the PDF.
            - 'score': The similarity score of the chunk to the query.
            - 'payload': The full payload if you need other metadata.
    """
    print(f"\n🔍 Retrieving documents from '{collection_name}' for query: \"{query}\"")

    # 1. Generate embedding for the query
    try:
        query_embedding = model.encode(query).tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    # 2. Search Qdrant for similar vectors
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,  # To retrieve the metadata and original text
            score_threshold=score_threshold # Optional: filter by score
        )
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []

    # 3. Process and return results
    results = []
    if search_results:
        for hit in search_results:
            payload = hit.payload
            results.append({
                "text": payload.get("text", ""), # The actual text chunk
                "source_pdf": payload.get("source_pdf", "N/A"),
                "chunk_number": payload.get("chunk_number", -1),
                "score": hit.score,
                "payload": payload # Include the full payload for flexibility
            })
        print(f"Found {len(results)} relevant chunks.")
    else:
        print("No relevant chunks found.")

    return results
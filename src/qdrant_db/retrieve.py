from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from strands import tool


QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = "https://4b6b6bd6-689d-4874-a9ab-f7f2489ee76b.us-east-1-0.aws.cloud.qdrant.io:6333"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
_qdrant_client = None
_embedding_model = None

def _initialize_global_clients():
    """Initializes global Qdrant client and embedding model."""
    global _qdrant_client, _embedding_model

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY environment variable not set. Cannot initialize Qdrant client.")
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL environment variable not set. Cannot initialize Qdrant client.")

    try:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        print(f"Global Qdrant client initialized successfully for URL: {QDRANT_URL}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize global Qdrant client: {e}")

    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Global SentenceTransformer model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load global SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}")

# Call initialization when the module is loaded.
# Consider implications if this script is part of a larger app (e.g., avoid re-initialization if not needed)
_initialize_global_clients()
    
@tool
def retrieve_relevant_texts(
    query: str,
    top_k: int = 5,
    score_threshold: float = None
) -> list:
    """
    Retrieves the most relevant text chunks from a Qdrant vector database
    based on the given text query. Uses globally configured Qdrant client,
    collection name, and sentence transformer model.

    Args:
        query (str): The user query to search in the knowledge base.
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
    global _qdrant_client, _embedding_model # Access global instances

    if _qdrant_client is None or _embedding_model is None:
        raise RuntimeError("Qdrant client or embedding model not initialized. Call _initialize_global_clients().")

    print(f"\n🔍 Retrieving documents from '{COLLECTION_NAME}' for query: \"{query}\"")

    # 1. Generate embedding for the query
    try:
        query_embedding = _embedding_model.encode(query).tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    # 2. Search Qdrant for similar vectors
    try:
        search_results = _qdrant_client.search(
            collection_name=COLLECTION_NAME,
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

if __name__ == '__main__':
    # This is an example of how to use it.
    # Ensure QDRANT_API_KEY is set as an environment variable before running.
    # Also, ensure your Qdrant instance at QDRANT_URL has the COLLECTION_NAME populated.
    print("Starting example usage of retrieve_relevant_texts...")
    if not QDRANT_API_KEY:
        print("Error: QDRANT_API_KEY environment variable is not set.")
        print("Please set it before running this example.")
        print("Example: export QDRANT_API_KEY='your_api_key_here'")
    else:
        try:
            # The clients are initialized when the module loads.
            # If you needed to re-initialize or ensure initialization:
            # _initialize_global_clients() # Though it's called above already.

            sample_query = "What is the main theme of the cosmos?"
            retrieved_docs = retrieve_relevant_texts(query=sample_query, top_k=3)

            if retrieved_docs:
                print(f"\n--- Top {len(retrieved_docs)} retrieved documents for query: \"{sample_query}\" ---")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\nDocument {i+1}:")
                    print(f"  Text: \"{doc['text'][:100]}...\"") # Print first 100 chars
                    print(f"  Source PDF: {doc['source_pdf']}")
                    print(f"  Chunk Number: {doc['chunk_number']}")
                    print(f"  Score: {doc['score']:.4f}")
            else:
                print(f"\nNo documents found for query: \"{sample_query}\"")

        except Exception as e:
            print(f"An error occurred during the example usage: {e}")
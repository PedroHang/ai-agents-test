import os
import uuid
import pdfplumber # For reading PDF files
from sentence_transformers import SentenceTransformer # For generating embeddings
from qdrant_client import QdrantClient, models

# --- Qdrant Client Initialization ---
# Ensure your QDRANT_API_KEY environment variable is set
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = "https://4b6b6bd6-689d-4874-a9ab-f7f2489ee76b.us-east-1-0.aws.cloud.qdrant.io:6333" # User's Qdrant URL

qdrant_client = None # Initialize to None
embedding_model = None # Initialize to None
EMBEDDING_DIMENSION = None # Initialize to None

if not QDRANT_API_KEY:
    print("Error: QDRANT_API_KEY environment variable not set.")
    exit()

try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    print("Successfully connected to Qdrant.")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    exit()

# --- Configuration for PDF Processing and Embeddings ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()
    print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded. Dimension: {EMBEDDING_DIMENSION}")
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
    exit()


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The concatenated text from the PDF, or None if an error occurs.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
        return full_text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size=512, overlap=50):
    """
    Splits text into smaller chunks.
    A simple implementation based on splitting by tokens (words) for now.
    More sophisticated methods (by sentences, paragraphs, or using NLP libraries) can be used.

    Args:
        text (str): The text to chunk.
        chunk_size (int): The approximate number of words per chunk.
        overlap (int): The number of words to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = min(current_pos + chunk_size, len(words))
        chunks.append(" ".join(words[current_pos:end_pos]))
        current_pos += (chunk_size - overlap)
        if current_pos >= len(words) and end_pos < len(words) and (len(words) - current_pos + chunk_size - overlap) < overlap : # ensure last chunk is captured
            chunks.append(" ".join(words[current_pos - (chunk_size - overlap) + (chunk_size - overlap) :]))
            break
        elif current_pos >= len(words):
            break

    # Ensure the very last part of the text is captured if the loop condition misses it
    if chunks and " ".join(words).endswith(chunks[-1].split()[-1]):
        pass
    elif len(words) > 0 :
        start_of_potential_last_chunk = max(0, len(words) - chunk_size)
        potential_last_chunk_text = " ".join(words[start_of_potential_last_chunk:])

        is_already_captured = False
        if chunks:
            last_captured_chunk_words = chunks[-1].split()
            potential_last_chunk_words = potential_last_chunk_text.split()
            if len(last_captured_chunk_words) >= len(potential_last_chunk_words):
                if last_captured_chunk_words[-len(potential_last_chunk_words):] == potential_last_chunk_words:
                    is_already_captured = True

        if not is_already_captured and potential_last_chunk_text:
            if not chunks or not chunks[-1].endswith(" ".join(words[-(chunk_size//2):])):
                chunks.append(potential_last_chunk_text)

    chunks = [chunk for chunk in chunks if chunk.strip()]
    return chunks


def upload_pdfs_to_qdrant(client: QdrantClient, pdf_directory: str, collection_name: str):
    """
    Processes PDF files from a directory, generates embeddings, and uploads them to Qdrant.

    Args:
        client (QdrantClient): An initialized Qdrant client.
        pdf_directory (str): Path to the directory containing PDF files.
        collection_name (str): Name of the Qdrant collection to use/create.
    """
    if not os.path.isdir(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found.")
        return

    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            print(f"Collection '{collection_name}' not found. Creating it...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Using existing collection '{collection_name}'.")
    except Exception as e:
        print(f"Error interacting with Qdrant collections: {e}")
        return

    points_to_upsert = []
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"\nProcessing PDF: {pdf_path}...")

            document_text = extract_text_from_pdf(pdf_path)
            if not document_text:
                print(f"No text extracted from {filename}. Skipping.")
                continue
            print(f"Extracted {len(document_text)} characters from {filename}.")

            text_chunks = chunk_text(document_text, chunk_size=256, overlap=30)
            if not text_chunks:
                print(f"No text chunks generated for {filename}. Skipping.")
                continue
            print(f"Split '{filename}' into {len(text_chunks)} chunks.")

            for i, chunk in enumerate(text_chunks):
                try:
                    vector = embedding_model.encode(chunk).tolist()
                    point_id = str(uuid.uuid4())
                    payload = {
                        "source_pdf": filename,
                        "chunk_number": i + 1,
                        "text": chunk,
                        "original_length_chars": len(chunk),
                    }
                    points_to_upsert.append(models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    ))
                    print(f"  Prepared chunk {i+1}/{len(text_chunks)} for {filename} (ID: {point_id})")
                except Exception as e:
                    print(f"Error encoding or preparing point for chunk {i+1} from {filename}: {e}")

            if points_to_upsert:
                try:
                    client.upsert(collection_name=collection_name, points=points_to_upsert)
                    print(f"Successfully upserted {len(points_to_upsert)} points from {filename} to '{collection_name}'.")
                    points_to_upsert = []
                except Exception as e:
                    print(f"Error upserting points to Qdrant for {filename}: {e}")
                    points_to_upsert = []

    print("\nPDF processing and uploading complete.")

def run_pdf_processing_pipeline(pdf_files_directory: str, qdrant_collection_name: str):
    """
    Main pipeline function to process PDFs and upload them to Qdrant.

    Args:
        pdf_files_directory (str): Path to the directory containing PDF files.
        qdrant_collection_name (str): Name of the Qdrant collection to use/create.
    """
    print("Starting PDF to Qdrant upload process...")

    # Ensure client and model are initialized (they are global in this script's context)
    if not qdrant_client or not embedding_model:
        print("Error: Qdrant client or embedding model not initialized. Exiting.")
        return

    # Create the PDF directory if it doesn't exist, for user convenience
    if not os.path.exists(pdf_files_directory):
        os.makedirs(pdf_files_directory)
        print(f"Created directory '{pdf_files_directory}'. Please add your PDF files there and re-run.")
        return # Changed exit() to return to allow calling this function multiple times if needed

    if not os.listdir(pdf_files_directory):
        print(f"The directory '{pdf_files_directory}' is empty. Please add PDF files to it and re-run.")
        return # Changed exit() to return

    # Call the main upload function
    upload_pdfs_to_qdrant(
        client=qdrant_client,
        pdf_directory=pdf_files_directory,
        collection_name=qdrant_collection_name
    )

    # You can verify by getting collection info
    try:
        collection_info = qdrant_client.get_collection(collection_name=qdrant_collection_name)
        print(f"\nCollection '{qdrant_collection_name}' info:")
        print(f"  Status: {collection_info.status}")
        print(f"  Points count: {collection_info.points_count}")
        print(f"  Vectors count: {collection_info.vectors_count}")
        print(f"  Config: {collection_info.config}")
    except Exception as e:
        print(f"Could not retrieve info for collection '{qdrant_collection_name}': {e}")


if __name__ == "__main__":
    # --- Configuration for the pipeline ---
    # IMPORTANT: Create a directory named 'my_pdfs' in the same location as this script,
    # and place your PDF files inside it. Or, change this path.
    PDF_DIR = "pdfs"
    COLLECTION_NAME = "pdf_documents_collection" # Choose a name for your collection

    run_pdf_processing_pipeline(
        pdf_files_directory=PDF_DIR,
        qdrant_collection_name=COLLECTION_NAME
    )
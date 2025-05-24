import os
from dotenv import load_dotenv
from .tools import *
from .agents import *
from .qdrant_db import *


if __name__ == "__main__":
    load_dotenv()
    os.getenv("QDRANT_HOST")
    os.getenv("QDRANT_PORT")
    os.getenv("QDRANT_API_KEY")

    # PDF_DIR = "pdfs"
    # COLLECTION_NAME = "pdf_documents_collection"

    # embed.run_pdf_processing_pipeline(
    #     pdf_files_directory=PDF_DIR,
    #     qdrant_collection_name=COLLECTION_NAME
    # )

    # Initialize the agent
    answer = letter_counter_agent.count_letters("I want you to count the number of times the letter 'a' appears in the word 'banana'.")

    print(answer)
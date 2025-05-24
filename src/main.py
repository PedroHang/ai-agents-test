import os
from dotenv import load_dotenv
from .tools import *
from .agents import *
from .qdrant_db import *

from strands import Agent, tool
from strands_tools import calculator, current_time, python_repl
from strands.models import BedrockModel


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

    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name='us-east-1',
        temperature=0.5,
    )

    agent = Agent(
        tools=[calculator, retrieve.retrieve_relevant_texts],
        model=bedrock_model,
        system_prompt="Você é um consultor de frases em livros."
    )


    # Initialize the agent
    answer = agent("Quero que consulte o livro cosmos e me diga o que acontece na parte que diz O desastre, qualquer que tenha sido, que retirou os dinossauros do mundo, removeu a pressão sobre os mamíferos.")

    print(answer)
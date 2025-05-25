import os
from dotenv import load_dotenv
from .tools import *
from .agents import *
from .qdrant_db import *

from strands import Agent
from strands_tools import calculator
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
        tools=[calculator, retrieve.retrieve_relevant_texts, plotly_agent.generate_plot],
        model=bedrock_model,
        system_prompt="Você é um agente responsável por garantir a resposta correta para a pergunta do usuário. Para isso, fará uso de ferramentas e agentes disponíveis"
    )


    # Initialize the agent
    answer = agent("Sobre o que é o documento em questão? em seguida gere um gráfico sobre o documento")

    print(answer)
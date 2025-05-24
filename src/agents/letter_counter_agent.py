from strands import Agent, tool
from strands_tools import calculator, current_time, python_repl
from strands.models import BedrockModel
from ..tools import *
from dotenv import load_dotenv

load_dotenv()

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name='us-east-1',
    temperature=0.5,
)

RESEARCH_ASSISTANT_PROMPT = """
You are an agent that is highly specialized in counting letter occurrences in words. you will count the exact
number of occurrences of a specific letter in a word and return only a single number, for example, if the word is "hello" and the letter is "l", you will return 2.
"""

@tool
def count_letters(query: str) -> str:
    try:
        agent = Agent(
            system_prompt=RESEARCH_ASSISTANT_PROMPT,
            model=bedrock_model,
            tools=[calculator, current_time, python_repl, letter_counter]
        )
        
        response = agent(query)
        return response
    except Exception as e:
        return f"An error occurred: {e}"
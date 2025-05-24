import os
from dotenv import load_dotenv
from .tools import *
from .agents import *


if __name__ == "__main__":
    load_dotenv()
    os.getenv("QDRANT_HOST")
    os.getenv("QDRANT_PORT")

    # Initialize the agent
    answer = letter_counter_agent.count_letters("I want you to count the number of times the letter 'a' appears in the word 'banana'.")

    print(answer)
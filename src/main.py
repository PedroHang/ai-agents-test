import os
from dotenv import load_dotenv
from tools import *
from agents.letter_counter_agent import letter_counter_agent


if __name__ == "__main__":
    load_dotenv()

    # Initialize the agent
    answer = letter_counter_agent.count_letters("I want you to count the number of times the letter 'a' appears in the word 'banana'.")

    print(answer)
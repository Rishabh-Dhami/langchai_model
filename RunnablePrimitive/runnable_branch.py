from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

parser = StrOutputParser()

# Prompt for happy jokes
happy_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a funny assistant."),
    ("human", "Tell me a light and cheerful joke.")
])
happy_chain = happy_prompt | model | parser

# Prompt for sad jokes
sad_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a thoughtful assistant."),
    ("human", "Tell me a dark or sarcastic joke.")
])
sad_chain = sad_prompt | model | parser

# Default chain
default_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a neutral assistant."),
    ("human", "Tell me a random joke.")
])
default_chain = default_prompt | model | parser

# RunnableBranch
joke_branch = RunnableBranch(
    (lambda x: x["mood"] == "happy", happy_chain),
    (lambda x: x["mood"] == "sad", sad_chain),
    default_chain
)

# Run
print(joke_branch.invoke({"mood": "happy"}))
print(joke_branch.invoke({"mood": "sad"}))
print(joke_branch.invoke({"mood": "confused"}))

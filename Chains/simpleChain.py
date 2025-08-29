from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate 5 interesting facts about the topic: \n {topic}")
])

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic" : "sex"})
print(result)
# chain.get_graph().print_ascii()
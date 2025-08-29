from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

template1 = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate a report about the topic: \n {topic}")
])

template2 = ChatPromptTemplate.from_messages([
     ("system", "You are the most helpfull assistent"),
    ("human", "Extract 5 intersting points from the report: \n  {report}")
])

parser = StrOutputParser()

chain = template1 | model | parser| template2 | model | parser

result = chain.invoke("Rag in llm")
print(result)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

prompt1 = ChatPromptTemplate([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate one linkdin post on this topic: {topic}")
])

prompt2 = ChatPromptTemplate([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate one tweet on this topic:  {topic}")
])

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : prompt1 | model | parser,
    'linkdin' : prompt2 | model | parser
})

result = parallel_chain.invoke({"topic": "rape"})
print(result)
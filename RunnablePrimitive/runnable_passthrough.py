from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate a joke on this topic : {topic}")
])

promt2 = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate a explaination about this joke : {joke}")
])

parser = StrOutputParser()

joke_chain = prompt1 | model | parser

runnable_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explanations" : (lambda X: {"joke": X}) | promt2 | model | parser
})

chain = joke_chain  | runnable_chain 

result = chain.invoke({"topic": "sex"})
print(result['explanations'])
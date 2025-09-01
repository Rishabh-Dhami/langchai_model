from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate a joke on this topic : {topic}")
])

def word_count(text):
    return len(text.split())

parser = StrOutputParser()


gen_joke_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "count" : RunnableLambda(word_count)
})

chain     = gen_joke_chain | parallel_chain

result = chain.invoke("hero")
print(result)
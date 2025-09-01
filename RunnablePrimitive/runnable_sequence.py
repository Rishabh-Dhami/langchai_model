from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

template = ChatPromptTemplate([
    ("system" , "you are a most helpfull assistent"),
    ("human", "Generate  only one joke on this topic : {topic}")
])


template1 = ChatPromptTemplate([
    ("system" , "you are a most helpfull assistent"),
    ("human", "Explain the follwing joke in hinglish : {text}")
])


parser = StrOutputParser()


runanle = RunnableSequence(template, model, template1, model, parser)

result = runanle.invoke({"topic": "Ram"})
print(result)
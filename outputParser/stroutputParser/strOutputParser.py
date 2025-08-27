from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

template1 = ChatPromptTemplate.from_messages([
    ("system", "You are the world's number 1 and most helpful assistant."),
    ("human", "Write a report on the topic:\n{topic}")
])

template2 = ChatPromptTemplate.from_messages([
    ("system" ,"You are the worlds number 1 and helpfull assistent."),
    ("human" , "wrtite a 5 line summery on the following text: \n {text}")
])

parser = StrOutputParser()



# Step 1: Report generator
report_chain = template1 | model | parser

# Step 2: Summarizer (takes the report as input)
summary_chain = template2 | model | parser

chain = report_chain | (lambda report: {"text": report}) | summary_chain

result = chain.invoke({"topic" : "c programming"})
print(result)




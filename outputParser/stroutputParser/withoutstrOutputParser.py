from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

template1 = ChatPromptTemplate.from_messages([
    ("system", "You are the world's number 1 and most helpful assistant."),
    ("human", "Write a report on the topic:\n{topic}")
])

template2 = ChatPromptTemplate  .from_messages([
    ("system" ,"You are the worlds number 1 and helpfull assistent."),
    ("human" , "wrtite a 5 line summery on the following text: \n {text}")
])


prompt1 = template1.format_messages(topic="Programming in c")



result = model.invoke(prompt1)
prompt2 = template2.format_messages(text=result.content)

result2 = model.invoke(prompt2)
print(result2.content)


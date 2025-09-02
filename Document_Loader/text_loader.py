from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=.7
)

parser = StrOutputParser()

prompt = ChatPromptTemplate([
    ("system" , "You are most helpfull assistent"),
    ("human" , "Generate a summary from this text: {text}")
])

chain = prompt | model | parser


loader = TextLoader("saas_dashboard_notes_utf8.txt",encoding="utf-8")
docs = loader.load()
result = chain.invoke({"text" : docs[0].page_content})
print(result)
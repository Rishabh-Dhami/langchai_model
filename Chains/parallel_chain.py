from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",   # choose another model if needed
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate short and simple notes from this text: \n {text}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Generate 5 questions answers from the following text : \n {text}")
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", "You are the most helpfull assistent"),
    ("human", "Merge the provided notes and quiz into a single document: \n notes -> {notes} \n questions answers -> {quiz}")
])

parser = StrOutputParser()


runnable_chain = RunnableParallel({
    "notes" : prompt1 | model1 | parser,
    "quiz" : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = runnable_chain | merge_chain

result = chain.invoke({"text" : """Artificial Intelligence (AI) is a branch of computer science that focuses on creating machines capable of performing tasks that normally require human intelligence. These tasks include problem-solving, understanding natural language, recognizing patterns, and making decisions. 

Machine Learning (ML), a subset of AI, enables systems to learn from data and improve performance without being explicitly programmed. For example, recommendation systems used by Netflix and Amazon are powered by ML algorithms that analyze user behavior. 

Another important field is Natural Language Processing (NLP), which allows machines to understand and generate human language. Applications of NLP include chatbots, language translation tools, and sentiment analysis systems. 

AI is increasingly being applied in healthcare, finance, transportation, and education. However, ethical concerns such as bias in algorithms, job displacement, and privacy issues must be addressed to ensure responsible use of AI.
"""})
print(result)



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="give the sentiment of the feedback")
    feedback: str = Field(description = "Include the original feedback.")

    
parser2 = PydanticOutputParser(pydantic_object=Feedback)    

template = ChatPromptTemplate.from_messages([
    ("system", "You are most helpfull assistent"),
    ("human", "Classify the sentiment of the following feedback text: {feedback} "
          "into Positive or Negative.\n"
          "The output should be in this format: {format_instructions} "
          "Also, include the original feedback.")

])

parser = StrOutputParser()

sentiment_chain = template | model | parser2




pos_prompt = ChatPromptTemplate([
    ("system", "You are the most helpful assistant"),
    ("human", 
     "The customer gave this positive feedback: {feedback}\n"
     "Write a single, concise, and appreciative response.\n"
     "Make it warm, professional, and customer-friendly.\n"
     "Do not provide multiple options, just one response."
    )
])

neg_prompt = ChatPromptTemplate([
    ("system", "You are the most helpful assistant"),
    ("human", 
     "The customer gave this negative feedback: {feedback}\n"
     "Write a single, empathetic, and professional response that acknowledges the concern.\n"
     "Do not provide multiple options, just one response."
    )
])



pos_chain = pos_prompt | model | parser
neg_chain  = neg_prompt | model | parser

branch_chain = RunnableBranch(
    ((lambda x: x["sentiment"] == "Positive"), pos_chain),
    ((lambda x: x["sentiment"] == "Negative"), neg_chain),
    RunnableLambda(lambda x : "could not fint sentiment" )  
)

chain = sentiment_chain  | RunnableLambda(lambda x: {"sentiment": x.sentiment, "feedback": x.feedback}) | branch_chain

feedback = "The product quality is amazing and delivery was super fast!"

result = chain.invoke({
    "feedback" : feedback,
    "format_instructions" : parser2.get_format_instructions()
})

print(result)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import json
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class Person(BaseModel):
    name : str = Field(description="Name of the friction place person")
    age : int = Field(gt=18, description="Age of the friction place person")
    city : str = Field("City of the friction place person")
    
class People(BaseModel):
    persons: List[Person] = Field(description="List of fictional people")    


template = ChatPromptTemplate.from_messages([
    ("system", "You are the number 1 and most helpfull assistent"),
    ("human", "Generate the name , age, city of the friction place {place} of 2 person\n\n"
     "Result should be in this format: {format_instructions}"
     )
    
])

parser = PydanticOutputParser(pydantic_object=People)

chain = template | model | parser
result = chain.invoke({"place" : "Australia", "format_instructions": parser.get_format_instructions()})

result_dict = result.dict()

print("dict", result_dict)

# ✅ Print dict as pretty JSON
# print(json.dumps(result_dict, indent=2))

# print(result.model_dump_json(indent=2))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Define the schema for output as an array
person_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the fictional person"},
            "gender": {"type": "string", "description": "Gender of the fictional person"},
            "age": {"type": "integer", "description": "Age of the fictional person"},
            "city": {"type": "string", "description": "City where the fictional person lives"}
        },
        "required": ["name", "gender", "age", "city"]
    }
}

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "You are the world's number one and most helpful assistant."),
    ("human", "Give me the gender, age, and city of a fictional person whose name is {name}. "
              "Return the output strictly as an ARRAY, even if there is only one object. "
              "Format: {format_instructions}")
])


# Parser
parser = JsonOutputParser(schema=person_schema)

# Chain
chain = template | model | parser

# Invoke
result = chain.invoke({
    "name": "Rishabh",
    "format_instructions": parser.get_format_instructions()
})
print("result:", result)

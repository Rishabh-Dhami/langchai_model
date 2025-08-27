from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

schema = [
    ResponseSchema(name="Fact_1", description="Fact  1 about the topic"),
    ResponseSchema(name="Fact_2", description="Fact  2 about the topic"),
    ResponseSchema(name="Fact_3", description="Fact  3 about the topic"),
]


template = ChatPromptTemplate.from_messages([
    ("system", "You are the world's number 1 and most helpful assistant."),
    ("human", "give best 3 fact about the topic:  {topic} \n\n"
     "Return them strictly in this format: {format_instructions}")
])



parser = StructuredOutputParser.from_response_schemas(schema)

raw_output = (template | model).invoke({
    "topic": "YouTube",
    "format_instructions": parser.get_format_instructions()
})
print("RAW MODEL OUTPUT:\n", raw_output)
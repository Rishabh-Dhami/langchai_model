from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",   # choose another model if needed
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

# Chat history as LangChain messages (not just strings!)
chat_history = [
    SystemMessage(content="You are a helpful assistant! Your name is Bob.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Add user message
    chat_history.append(HumanMessage(content=user_input))
    
    # Get model response (using invoke on entire chat history)
    result = model.invoke(chat_history)
    
    # Add AI response to history
    chat_history.append(AIMessage(content=result.content))
    
    print("Bob:", result.content)

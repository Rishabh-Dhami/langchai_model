from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()


def create_model():
    """
    Initialize the HuggingFace chat model.
    """
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.7,
    )
    return ChatHuggingFace(llm=llm)


def create_prompt():
    """
    Create a simple chat prompt template.
    """
    return ChatPromptTemplate(
        ("system", "You are a helpful expert in {domain}. "),
        ("human", "Explain in simple terms: what is {topic}?")
    )


def run_chat(domain: str, topic: str):
    """
    Run the chat model with given domain and topic.
    """
    model = create_model()
    chat_template = create_prompt()
    chain = chat_template | model

    result = chain.invoke({"domain": domain, "topic": topic})
    return result.content


if __name__ == "__main__":
    print("Welcome to the AI Expert Assistant! Type 'exit' to quit.\n")

    while True:
        domain = input("Enter a domain (e.g., Physics, History, Medicine): ").strip()
        if domain.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        topic = input("Enter a topic: ").strip()
        if topic.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        response = run_chat(domain, topic)
        print("\nAI:", response, "\n")

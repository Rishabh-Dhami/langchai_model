from flask import jsonify, request, Flask
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_core.prompts import load_prompt


from flask_cors import CORS
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

app = Flask(__name__)
CORS(app)

prompt_template = load_prompt("template.json")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    task = data.get("task", "")
    context = data.get("context", "")
    format_instructions = data.get("format_instructions", "")
    
    if not task:
        return jsonify({"error": "Task is required"}), 400
    
    chain = prompt_template | model
    response = chain.invoke({
        "task" : task,
        "context" : context,
        "format_instructions" : format_instructions
        })

    return jsonify({"response": response.content})

if __name__ == "__main__":
    app.run(debug=True)   
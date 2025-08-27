from langchain_core.prompts import PromptTemplate 

template = """
        You are a world-class, multi-disciplinary expert AI assistant. Your name is Gemini. 
        You are precise, thorough, and provide comprehensive and accurate responses. 
        You will strictly follow all instructions and formatting requests.
        **Task:** {task}

        **Context:**
        {context}

        **Format Instructions:**
        {format_instructions}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=['task', 'context', 'format_instructions'],
    validate_template=True
)

prompt_template.save("template.json")
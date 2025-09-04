from langchain.text_splitter import CharacterTextSplitter

# text = """
# The Future of Artificial Intelligence

# Artificial Intelligence (AI) has moved from being a futuristic idea in science fiction to an everyday reality shaping how we live, work, and interact. Over the past decade, advances in machine learning, natural language processing, and computer vision have made AI systems more powerful, reliable, and accessible. From personal assistants like Siri and Alexa to recommendation engines on YouTube and Netflix, AI is embedded in our daily routines, often in ways we do not consciously recognize.

# One of the most significant promises of AI is its ability to automate repetitive tasks. For instance, industries such as healthcare are now using AI to assist doctors in analyzing medical images, predicting diseases, and personalizing treatment plans. Similarly, financial institutions rely on AI to detect fraud and improve investment decisions. Beyond business, AI plays an important role in climate modeling, helping scientists forecast weather patterns, analyze environmental data, and suggest solutions for reducing carbon emissions.

# However, the rise of AI also brings challenges. Ethical concerns about bias, transparency, and accountability in AI decision-making have grown louder. For example, if an algorithm used in hiring favors certain groups over others, it can reinforce social inequalities rather than solve them. Governments and organizations are now debating how to regulate AI systems responsibly without stifling innovation. Data privacy is another major issue, as AI thrives on massive datasets that often contain sensitive personal information.

# Looking ahead, the future of AI seems both promising and uncertain. On one hand, technologies like autonomous vehicles, intelligent robots, and generative AI could revolutionize entire industries. On the other hand, the risk of job displacement, misuse of AI in surveillance, and creation of deepfakes raises legitimate concerns. The key lies in finding a balance—leveraging AI for human progress while ensuring fairness, transparency, and ethical responsibility. Ultimately, AI is not just about machines becoming smarter; it is about how humanity chooses to shape and use this powerful tool for a better tomorrow.
# """

# splitter = CharacterTextSplitter(
#     chunk_size= 100,
#     chunk_overlap=0,
#     separator=' '
# )

# result = splitter.split_text(text=text)
# print(result)

from langchain_community.document_loaders import PyPDFLoader

file_url = r"C:\Users\91893\Projects\langchai_model\Sample\sample1.pdf"
loader = PyPDFLoader(file_path=file_url)
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_documents(docs)

print(result[0])
print(result[1])
from langchain_community.document_loaders import  WebBaseLoader

url = "https://en.wikipedia.org/wiki/Natural_language_processing"
loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0].page_content[:500])  # print first 500 chars

from langchain_community.document_loaders import PyPDFLoader

# Use absolute path
file_path = r"C:\Users\91893\Projects\langchai_model\Document_Loader\RAG_Guide.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

# Print first page content
print(len(docs))
print(docs[0].metadata)
print(docs[0].page_content)

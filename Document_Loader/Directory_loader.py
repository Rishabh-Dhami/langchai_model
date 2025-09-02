from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

folder_path = r"C:\Users\91893\Projects\langchai_model\Sample"

txt_loader = DirectoryLoader(
    path=folder_path,
    glob="*.txt",
    loader_cls=TextLoader
)

# PDF loader
pdf_loader = DirectoryLoader(
    path=folder_path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = txt_loader.load() + pdf_loader.load()

# docs = list(txt_loader.lazy_load())
print(docs)

print(docs)
from langchain_community.document_loaders import CSVLoader

file_path = r"C:\Users\91893\Projects\langchai_model\Document_Loader\csv_loader_data.csv"
loader = CSVLoader(file_path=file_path)
docs = loader.load()

for doc in docs:
    print(doc)

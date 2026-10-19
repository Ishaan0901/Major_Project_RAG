from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import model
from dotenv import load_dotenv
load_dotenv()


#   Document Loaders:
docs=PyPDFDirectoryLoader('D:/major project/data/RAG_Pdfs').load()



#   Text Splitters:
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)



#   VectorStore:
FAISS.from_documents(
    embedding=model,
    documents=chunks
).save_local('Vector_Store')
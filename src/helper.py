from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceBerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import os
#now to create function to extract pdf files 
def load_pdf_file(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents=loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]: 
    """Given a list of documents objects , return a new list of Documents objects containing only 'source' 
     in metadata and the original page_content.
     """
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata ={"source":src}
            )
        )
    return minimal_docs 

#therefore now lets start with chunking of the data 
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk= text_splitter.split_documents(minimal_docs)
    return text_chunk

 
def download_embedding():
   """
   Download and return the HuggingFaceEmbedding Models
   """
   model_name="sentence-transformers/all-MiniLM-L6-v2"
   embedding=HuggingFaceEmbeddings(
      model_name = model_name,
   )
   return embedding  
embedding=download_embedding()

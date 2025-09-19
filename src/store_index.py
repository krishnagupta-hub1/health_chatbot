from dotenv import load_dotenv
import os
from src.helper import load_pdf_file,filter_to_minimal_docs,text_split,download_embedding
from pinecone import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#HUGGINGFACEHUB_API_TOKEN= os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY
#os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACEHUB_API_TOKEN

extracted_data = load_pdf_file(r"C:\Users\Krishna Gupta\Desktop\health_chatbot\data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk=text_split(minimal_docs)
embedding = download_embedding()

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

print("existing indexes:", pc.list_indexes())

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,   #dimension in embedding
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

#embed each chunk and upsert the embedding into your pinecone index 
docsearch = PineconeVectorStore.from_existing_index(
     embedding = embedding,
     index_name = index_name 
)

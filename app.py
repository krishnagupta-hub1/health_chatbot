from flask import Flask, render_template,jsonify,request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import HuggingFaceHub
#from langchain_huggingface import HuggingFaceHub

from langchain_community.chat_models import ChatHuggingFace

from dotenv import load_dotenv
from src.prompt import *

import os 

app= Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#HUGGINGFACEHUB_API_TOKEN= os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY
#os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACEHUB_API_TOKEN

embedding = download_embedding()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
     embedding = embedding,
     index_name = index_name 
)

retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
#gives llm based answer in vector form 
  
chatModelLLM = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

#repo_id = "medalpaca/medalpaca-7b"

#hf_llm=HuggingFaceHub(
   # repo_id = repo_id,
   # huggingfacehub_api_token=OPENAI_API_KEY,
   # model_kwargs={"temperature":0.0,"max_new_tokens":512}
#)

#chatModelLLM=ChatHuggingFace(llm=hf_llm)

#main chaining part 
question_answer_chain = create_stuff_documents_chain(chatModelLLM,prompt)
rag_chain = create_retrieval_chain(retriever , question_answer_chain)
#now answer from vector format --> word format (user understanding)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug = True)

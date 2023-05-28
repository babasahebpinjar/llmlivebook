import os
import pinecone
import os
import getpass
import pinecone 
from tqdm.autonotebook import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from fastapi import FastAPI, File, UploadFile, HTTPException

OPENAI_KEY = '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'
chatHistory = 0
vectorstore = ''
qa = ''
index_created = 0
os.environ['OPENAI_API_KEY'] = OPENAI_KEY


def getDocuments(folderName):
    
    files = folderName +'/'
    pdf_loader = DirectoryLoader(files, glob="**/*.pdf")
    readme_loader = DirectoryLoader(files, glob="**/*.md")
    txt_loader = DirectoryLoader(files, glob="**/*.txt")
    #take all the loader
    loaders = [pdf_loader, readme_loader, txt_loader]

    #lets create document 
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
        
    print (f'You have {len(documents)} document(s) in your data')
    print (f'There are {len(documents[0].page_content)} characters in your document')

    #Split the Text from the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40) #chunk overlap seems to work better
    documents = text_splitter.split_documents(documents)
    print(len(documents))

    return documents
#Embeddings and storing it in Vectorestore

def getVectoreStore(documents):
    global vectorstore
    chromaVector = 1
    pineconeVector = 0
    loadExistingPineConeIndex = 0
    
    if chromaVector == 1:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)        
        vectorstore = Chroma.from_documents(documents, embeddings)
        index_created = 1

    # pinecone
    if pineconeVector == 1:
    
        PINECONE_API_KEY = getpass.getpass('Pinecone API Key:')
        PINECONE_ENV = getpass.getpass('Pinecone Environment:')

        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_ENV  # next to api key in console
        )

        index_name = "langchain-demo"
        vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name)


        if loadExistingPineConeIndex == 1:

            # if you already have an index, you can load it like this           

            # initialize pinecone
            pinecone.init(
                api_key=PINECONE_API_KEY,  # find at app.pinecone.io
                environment=PINECONE_ENV  # next to api key in console
            )

            index_name = "langchain-demo"
            vectorstore = Pinecone.from_existing_index(index_name, embeddings)

    return vectorstore


def getSimilarDocument(query):
    global vectorstore
    query = "Who are the authors of gpt4all paper ?"
    docs = vectorstore.similarity_search(query)

    return docs

    #print(docs[0].page_content)


def createLangchainQA(vectorstore):
    global chatHistory,qa
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
    llm = OpenAI(model_name="gpt-3.5-turbo",openai_api_key=OPENAI_KEY,temperature =0)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever)
    
    return qa
    #qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever)
    #llm = OpenAI(model_name="gpt-3.5-turbo",openai_api_key=OPENAI_KEY)
    # chat_history = []
    # #query = "How much is spent for training the gpt4all model?"
    # result = qa({"question": query, "chat_history": chat_history})
    # result["answer"]

    # if chatHistory == 1:
    #     chat_history.append((query, result["answer"]))
    #     chat_history

# query = "What is this number multiplied by 2?"
# result = qa({"question": query, "chat_history": chat_history})
# result["answer"]
from flask import Flask, Response , request
import requests
app = Flask(__name__)

#@app.route("/qaonfiles")
@app.route("/qaonfiles", methods=["GET", "POST"])
def qaonfiles():  
    if request.method == "GET":
        query = request.args.get("query")  
        chat_history = []
        #query = "How much is spent for training the gpt4all model?"
        result = qa({"question": query, "chat_history": chat_history})
        return result["answer"]
    else:
        return "Only get method"

import shutil
import atexit
def cleanup():
    print("Cleaning up before exit...")
    if index_created == 1:
        vectorstore = vectorstore.vectorstore            
        vectorstore.delete_collection()
    if os.path.exists('.chroma'):        
        directory = ".chroma"
        shutil.rmtree(directory) 
    # Add your code to be executed here

atexit.register(cleanup)

# Start the server on port 3000
if __name__ == "__main__":    
    documents = getDocuments('files')
    vectorstore = getVectoreStore(documents)
    qa = createLangchainQA(vectorstore)
    app.run(port=3000)
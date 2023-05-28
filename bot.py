from flask import Flask, Response
from slackeventsapi import SlackEventAdapter
import os
from threading import Thread
from slack import WebClient
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader


# This `app` represents your existing Flask app
app = Flask(__name__)

greetings = ["hi", "hello", "hello there", "hey"]

#SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET']
#SLACK_SIGNING_SECRET = '25785962d541d481356de4dbae2ef9fb' # sherpababatest
#SLACK_SIGNING_SECRET = 'e6c787153f4c46b4c2517dc1665ab0b6' #sherpaabc
SLACK_SIGNING_SECRET = '94bc69180c42a755eb67407a4789fc68' #sherpaasic


#slack_token = os.environ['SLACK_BOT_TOKEN']
#slack_token = 'xoxb-5304562976659-5317485284017-WhfdpIWqWGx3p9WHkOic5qNU' #sherpababatest
#slack_token = 'xoxb-4640689820821-5332468726450-hT9Fm73nxg72YEee4Hq2CAvw' #sherpaabc
slack_token = 'xoxb-337418436452-5329680352485-U3ShgpknLtZpzzxVb7OTmRSL' #sherpaasic

#VERIFICATION_TOKEN = os.environ['VERIFICATION_TOKEN']
#VERIFICATION_TOKEN = 'ovSINFVeFOdJ6YSaCjjIbuSP' # sherpababatest
#VERIFICATION_TOKEN = 'lpXG1aLKCBxivBLvtmnoy5O9' #sherpaabc
VERIFICATION_TOKEN = '1pqOhlNbgArHTAWTPORdDFFw'

#instantiating slack client
slack_client = WebClient(slack_token)

#OPENAI_KEY = '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' # baba api key : Free quota over
OPENAI_KEY = ''  # John's api-key, don't abuse!
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

# An example of one of your Flask app's routes
@app.route("/")
def event_hook(request):
    json_dict = json.loads(request.body.decode("utf-8"))
    if json_dict["token"] != VERIFICATION_TOKEN:
        return {"status": 403}

    if "type" in json_dict:
        if json_dict["type"] == "url_verification":
            response_dict = {"challenge": json_dict["challenge"]}
            return response_dict
    return {"status": 500}
    return


slack_events_adapter = SlackEventAdapter(
    SLACK_SIGNING_SECRET, "/slack/events", app
)  

chat_history = []
@slack_events_adapter.on("app_mention")
def handle_message(event_data):
    global chat_history
    chat_history = []
    def send_reply(value):
        event_data = value
        message = event_data["event"]
        if message.get("subtype") is None:
            command = message.get("text")
            
            print('-------',command)
            channel_id = message["channel"]
            
            # C056RU8PEJ0 : llm live project
            # C059L0UFLR4 : bottest
            allowed_channels = ["C056RU8PEJ0", "C059L0UFLR4"]  # Add the desired channel IDs
            if channel_id not in allowed_channels:
                print("Not allowed to process")
                return
            #if any(item in command.lower() for item in greetings):
            #result = qa({"question": command, "chat_history": []})
            #message= (result["answer"])
            message = ("Helloooooooo <@%s>! :tada: I am good how are you" % message["user"] )
            
            print(channel_id)
            global chain
            message = chain.run(command)
            print(message)
            slack_client.chat_postMessage(channel=channel_id, text=message)
            command = ''
    thread = Thread(target=send_reply, kwargs={"value": event_data})
    thread.start()
    return Response(status=200)



#@app.post('/createVectorStore', response_class=ORJSONResponse)
# @app.route("/createVectorStore")
# def createVectorStore(transcript: UploadFile = File(...)):
#     if  transcript.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed")
           
#     with open(transcript.filename, mode="wb") as new_file:
#         new_file.write(transcript.file.read())
    
#     new_file.close()
#     doc_reader = PdfReader(transcript.filename)    
#     # read data from the file and put them into a variable called raw_text
#     raw_text = ''
#     for i, page in enumerate(doc_reader.pages):
#         text = page.extract_text()
#         if text:
#             raw_text += text            

#     # Splitting up the text into smaller chunks for indexing
#     text_splitter = CharacterTextSplitter(        
#         separator = "\n",
#         chunk_size = 1000,
#         chunk_overlap  = 200, #striding over the text
#         length_function = len,
#     )
#     texts = text_splitter.split_text(raw_text)

#     # Download embeddings from OpenAI
#     embeddings = OpenAIEmbeddings()
#     docsearch = FAISS.from_texts(texts, embeddings)

#     #extracted_list = []
#     chain = load_qa_chain(OpenAI(), 
#                       chain_type="stuff") # we are going to stuff all the docs in at once
    
#     query = "hi"
#     docs = docsearch.similarity_search(query)
#     result = chain.run(input_documents=docs, question=query)    
    
    
#     return result
######################################

### Get code from John ot create index

######################################


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





####################
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

loaders = ''
chain = ''
index = ''
def createIndex(pdf_folder_path):
    files = os.listdir(pdf_folder_path)

    global OPENAI_KEY
    # Print the list of files
    print(files)
    global loaders
    global chain
    global index
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    #loaders
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
        
    index = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_KEY),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

    llm = OpenAI(model_name="gpt-3.5-turbo",openai_api_key=OPENAI_KEY)#, n=2, best_of=2)
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type="stuff", 
                                        retriever=index.vectorstore.as_retriever(), 
                                        input_key="question")
    
    return chain

# Start the server on port 3000
if __name__ == "__main__":
    # documents = getDocuments('files')
    # vectorstore = getVectoreStore(documents)
    # qa = createLangchainQA(vectorstore)
    
    chain = createIndex("files")
    app.run(port=3000)
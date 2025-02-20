from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle
import pandas as pd
import numpy as np
import sentence_transformers
from langchain import PromptTemplate
from urllib.parse import urljoin
#from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationBufferWindowMemory
from fastapi.middleware.cors import CORSMiddleware
from langchain import PromptTemplate
import torch
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from llama_cpp import Llama
from datetime import datetime
from time import time
import logging
import base64
from fastapi import BackgroundTasks
from logging.handlers import TimedRotatingFileHandler

use_cuda = torch.cuda.is_available()
index = torch.cuda.current_device()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

device = torch.device('cuda:'+ str(index) if use_cuda else 'cpu')
print("Device: ",device)

# here we set the level,message and handler for the our log file
logger = logging.getLogger('Api logs')  # name of class
logging.basicConfig(level=logging.DEBUG)  # we set config level to debug
current_date = datetime.now().strftime('%Y-%m-%d')
log_file_name = f'/home/sachin/RON/ZB_Bank/logs/logger_file_{current_date}.log'
log_handler = TimedRotatingFileHandler(log_file_name, when='midnight', interval=1, backupCount=0)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')  # here we set the format for message.
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)


#device = torch.device('cuda:0')  # Use the appropriate GPU device ID
torch.cuda.set_device(device)

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Initialize app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

greet = ["hey","hello","hi","hello there","good morning","good evening","hey there","let's go","hey dude","goodmorning","goodevening","good afternoon","hii","good bye","see you later","good night","bye","goodbye","have a nice day","see you around","bye bye","bye see you","yes","y","of course","that sounds good","correct","no","n","never","I don't think so","don't like that","no way","not really","perfect","great","amazing","feeling like a king","wonderful","I am feeling very good","I am great","I am amazing","I am going to save the world","super stoked","extremely good","so so perfect","so good","so perfect","my day was horrible","I am sad","I don't feel very well","I am disappointed","super sad","I'm so sad","sad","very sad","unhappy","not good","not very good","extremely sad","so sad","are you a bot?","are you a human?","am I talking to a bot?","am I talking to a human?","thanks","thank you","OK thanks","Thanks","many thanks","thanks a lot","thank you so much","thank u very much"]

# Define the input model
class SearchInput(BaseModel):
    query: str
    user_name: str

# Define the input model
class DocInput(BaseModel):
    query: str
    user_name: str
    file_data:str


#convert Base 64 to PDF
def base64_to_pdf(base64_data, output_path):
    # Decode the base64 string
    pdf_data = base64.b64decode(base64_data)
    # Write the decoded data to a PDF file
    with open(output_path, 'wb') as pdf_file:
        pdf_file.write(pdf_data)
        
             
def generate_pdf_path(user_id, pdf_name="output.pdf", base_folder="Document"):
    # Create a folder based on the user ID
    user_folder = os.path.join(base_folder, f"{user_id}")
    os.makedirs(user_folder, exist_ok=True)

    # Dynamic output path for the PDF file
    output_path = os.path.join(user_folder, pdf_name)

    return output_path
 
# Load PDF documents using PyPDFLoader
def load_pdf_documents(pdf_folder_path):
    pdf_documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            pdf_documents.extend(loader.load())
        elif file.endswith(".csv"):
            csv_path = os.path.join(pdf_folder_path, file)
            loader=CSVLoader(csv_path,encoding='cp1252')
            pdf_documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path=data_path+file
            loader=TextLoader(text_path)
            document.extend(loader.load())
    if len(pdf_documents) == 0:
        return None
    return pdf_documents
    
def load_single_document(doc_path):
    documents = []
    if os.path.isfile(doc_path):
        if doc_path.endswith(".pdf"):
            loader = PyPDFLoader(doc_path)
        elif doc_path.endswith(".csv"):
            loader = CSVLoader(doc_path, encoding='cp1252')
        elif doc_path.endswith(".txt"):
            loader = TextLoader(doc_path)
        documents.extend(loader.load())
    else:
        return None
    return documents
    

# Load Faiss vector database
def load_faiss_vector_database(docs, embeddings, db_path):
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_path)
    return db

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')


# loading main db
main_db_directory = "/home/sachin/RON/ZB_Bank/vectorstore/db_faiss"
main_db = FAISS.load_local(main_db_directory, embeddings)

# Load Faiss vector database
def load_main_db():
    return FAISS.load_local(main_db_directory, embeddings)

# Background task to reload the vector database
def reload_vector_database():
    global main_db
    main_db = load_main_db()



# Callback manager for LLM
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LLM configuration
n_gpu_layers = 40
n_batch = 256
llm = LlamaCpp(
        model_path=model_path,  # Ensure model_path is defined
        max_tokens=2048,
        temperature=0.05,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        n_ctx=4096,
        verbose=False
      )

template = """
[INST] <<SYS>>
You are Ron, an expert on ZB Financial Holdings Limited. You're here to share insights about financial services, covering commercial banking, leasing, trust, and executor services.
<</SYS>>
{context}
Question: {query}
Helpful Answer:
[/INST]
"""

prompt = PromptTemplate(
    input_variables=["query","context"], template=template
)

chain = load_qa_chain(
    llm, chain_type="stuff",prompt=prompt
)


@app.post('/Ron')
async def search_results(request: SearchInput, background_tasks: BackgroundTasks):
    try:
        startTime = time()
        logger.info("Entry from search_results Method")
        logger.info(f"User Name: {request.user_name}")
        # Reload the vector database in the background
        #reload_vector_database()
        # Perform search and get response
        query = request.query.lower()
        logger.info(f"Question: {query}")
        docs=main_db.similarity_search(query)
        print("###########")
        print("Context",docs)
        #chat_history = chain.memory.buffer or ""  # Corrected method name
        response = chain({"input_documents": docs, "query": query}, return_only_outputs=True)
        source_info = []
        if query not in greet:
            for result in docs:
                file_path = result.metadata['source']
                file_path_parts = file_path.split("/")
                pdf_name = file_path_parts[-1]
                # Create a web URL by joining the base URL and PDF file name
                base_url = "https://www.zb.co.zw/"
                #web_url = urljoin(base_url)
                source_info.append({'link': base_url, 'name': pdf_name})
        unique_links = set()
        unique_source_info = []
        for item in source_info:
            link = item["link"]
            if link not in unique_links:
                unique_links.add(link)
                unique_source_info.append(item)
        print("unique_source_info",unique_source_info)
        logger.info(f"Response:{response}, source_info:{unique_source_info}")
        stopTime = time()
        logger.info("Exit from search_results Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        return {"response": response,"source_info": unique_source_info}
    
    except Exception as e:
        logger.error(f"An error occurred while search_results responce. Details : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Function to handle PDF processing and database creation
def process_pdf_and_create_db(request):
    try:
        user_name = request.user_name
        user_temp_folder = f'/home/sachin/RON/ZB_Bank/Temp/vectorstore/{user_name}'
        pdf_path = generate_pdf_path(user_name)
        print(pdf_path)
        logger.info(f"User Name: {request.user_name}")
        print(request.file_data)
        if request.file_data:
            base64_to_pdf(request.file_data, pdf_path)
            pdf_dir = f"{pdf_path.split('/')[0]}/{pdf_path.split('/')[1]}"
            pdf_documents = load_pdf_documents(pdf_dir)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=50)
            docs = text_splitter.split_documents(pdf_documents)

            # Create user-specific temp folder if not exists
            if not os.path.exists(user_temp_folder):
                os.makedirs(user_temp_folder)

            Temp_db = load_faiss_vector_database(docs, embeddings, f'{user_temp_folder}/db_faiss')
            logger.info("File uploaded successfully")
        else:
            # Check if the database already exists
            db_path = f'{user_temp_folder}/db_faiss'
            if os.path.exists(db_path):
                Temp_db = FAISS.load_local(db_path, embeddings)
                logger.info("Loaded existing database")
            else:
                logger.info("No new PDF or existing database found")
                return None  # No PDF or database, return None

        return Temp_db,docs

    except Exception as e:
        logger.error(f"Error processing PDF and creating database: {str(e)}")
        return None


#####################################################################################################

prompt_Document_Training = """
[INST] <<SYS>>
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
<</SYS>>
{context}
Question: {question}
[/INST]
"""

@app.post('/Document_Training')
async def document_training(request: DocInput):
    try:
        startTime = time()
        logger.info("Entry from Document_Training Method")
        logger.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        question = request.query.lower()
        logger.info(f"Question: {question}")
        prompt = PromptTemplate(input_variables = ["context", "question"], template=prompt_Document_Training)
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(question)
        response = llmChain({"input_documents": docs, "question": question}, return_only_outputs=True)
        logger.info(f"Response:{response}")

        stopTime = time()
        logger.info("Exit from Document_Training Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        return {"response": response}

    except Exception as e:
        logger.error(f"Error in Document_Training method: {str(e)}")
        return {"response": "Internal server error"}


#####################################################################################################

promptTemplateTextSummerization = """
[INST] <<SYS>>
Please summarize the context provided below in a clear and concise manner, highlighting the key points and main ideas. 
<</SYS>>
{context}
Question: {question}
[/INST]
"""
    
@app.post('/Summarization')
async def Summarization(request: DocInput):
    try:
        startTime = time()
        logger.info("Entry from Summarization Method")
        logger.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        question = "What are the main takeaways from the document, and can you provide a brief summary of its key points?"
        logger.info(f"Question: {question}")
        prompt = PromptTemplate(input_variables = ["context", "question"], template=promptTemplateTextSummerization)
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(question)
        response = llmChain({"input_documents": docs, "question": question}, return_only_outputs=True)
        logger.info(f"Response:{response}")

        stopTime = time()
        logger.info("Exit from Summarization Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        return {"response": response}

    except Exception as e:
        logger.error(f"Error in Summarization method: {str(e)}")
        return {"response": "Internal server error"}
        

#######################################################################################################
# Replace "{entity}" with the desired entities and set the question
NER_prompt = """
[INST] <<SYS>>
You're in charge! Specify the Named Entities you're interested in (e.g., person names, locations, organizations, dates, times, Amounts, Account Numbers, Transaction Details,Interest Rates,Loan Terms, Financial Products, etc.), and I'll extract them for you. Keep your queries safe, respectful, and positive.
<</SYS>>
{context}
Question: {question}
Entities of Interest: person names, locations, organizations, dates, times,  Amounts, Account Numbers,Transaction Details,Interest Rates, Loan Terms, Financial Products,  etc.
Other Entities:
[/INST]
"""

@app.post('/NER')
async def ner(request: DocInput):
    try:
        startTime = time()
        logger.info("Entry from ner Method")
        logger.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        question = "extract entity from this document"
        logger.info(f"Question: {question}")
        prompt = PromptTemplate(input_variables = ["context", "question"], template=NER_prompt)
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(question)
        response = llmChain({"input_documents": docs, "question": question}, return_only_outputs=True)
        logger.info(f"Response:{response}")

        stopTime = time()
        logger.info("Exit from ner Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        return {"response": response}

    except Exception as e:
        logger.error(f"Error in ner method: {str(e)}")
        return {"response": "Internal server error"}
        
        
##################################################################################################

promptTemplateClassification = """
[INST] <<SYS>>
You're in charge! Classify the following context into one of the industry categories: Insurance, Technology, Healthcare, Finance, Retail, Manufacturing, Education, Energy, Entertainment, Agriculture, Transportation, Automobile, Food industry . Provide a industry label for the most appropriate industry based on the context.
Reply with most three relevant industries categories: Insurance, Technology, Healthcare, Finance, Retail, Manufacturing, Education, Energy, Entertainment, Agriculture, Transportation, Automobile, Food industry.
<</SYS>>
{context}
Question: {question}
[/INST]
"""


@app.post('/Classification')
async def classification(request: DocInput):
    try:
        startTime = time()
        logger.info("Entry from classification Method")
        logger.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        question = "classify this document for me"
        logger.info(f"Question: {question}")
        prompt = PromptTemplate(input_variables = ["context", "question"], template=promptTemplateClassification)
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(question)
        response = llmChain({"input_documents": docs, "question": question}, return_only_outputs=True)
        logger.info(f"Response:{response}")

        stopTime = time()
        logger.info("Exit from classification Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        return {"response": response}

    except Exception as e:
        logger.error(f"Error in classification method: {str(e)}")
        return {"response": "Internal server error"}

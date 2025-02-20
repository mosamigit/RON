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
import shutil
import configparser
import requests
import psutil
import platform
from datetime import time, timedelta

from pymongo import MongoClient



use_cuda = torch.cuda.is_available()
index = torch.cuda.current_device()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

device = torch.device('cuda:'+ str(index) if use_cuda else 'cpu')
print("Device: ",device)

logging.basicConfig(filename='logs/ALL_RON_Logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


#device = torch.device('cuda:0')  # Use the appropriate GPU device ID
torch.cuda.set_device(device)

# Config Path
config_path = "RON_config.ini"

# Read from config file
get_section = configparser.ConfigParser(interpolation=None)
get_section.read(config_path)
model_name_or_path = get_section.get('lama_model_name', 'model_name_or_path')
model_basename = get_section.get('lama_model_name', 'model_basename') # the model is in bin format
#pdf_documents_path = get_section.get('pdf_documents', 'pdf_documents_path')
chunk_size = get_section.getint('pdf_documents', 'chunk_size')
chunk_overlap = get_section.getint('pdf_documents', 'chunk_overlap')
embedding_model_name = get_section.get('embedding_model_name', 'embedding_model_name')
main_db_directory = get_section.get('faiss_vactor_db', 'ron_main_vector_db_directory')
max_token = get_section.get('lama_model_conf', 'max_token')
temperature = get_section.getint('lama_model_conf', 'temperature')
n_gpu_layers = get_section.getint('lama_model_conf', 'n_gpu_layers')
n_batch = get_section.getint('lama_model_conf', 'n_batch')
n_ctx = get_section.getint('lama_model_conf', 'n_ctx')
verbose = get_section.getboolean('lama_model_conf', 'verbose')
actual_pdf_base_url = get_section.get('actual_pdf', 'actual_pdf_base_url')
#user_temp_folder = get_section.get('temp_vector_db', 'user_temp_folder')
template = get_section.get('RON_lama_system_prompt', 'DEFAULT_SYSTEM_PROMPT')
promptTemplateTextSummerization = get_section.get('RON_lama_system_prompt', 'promptTemplateTextSummerization')
NER_prompt = get_section.get('RON_lama_system_prompt', 'NER_prompt')
promptTemplateClassification = get_section.get('RON_lama_system_prompt', 'promptTemplateClassification')
prompt_Document_Training = get_section.get('RON_lama_system_prompt', 'prompt_Document_Training')


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
    file_name:str
    file_data:str


# MongoDB configuration
mongo_client = MongoClient("mongodb://viren:viren$456@172.17.2.94:27017/ron_stage")
db = mongo_client["ron_stage"] 
log_collection = db["ai_ml_logs"]


# Function to log events into MongoDB
def log_to_mongo(user_name,level, message, time,api_name, source_name="ai ml api", additional_data=None):
    log_entry = {
        "user_name":user_name,
        "source_name": source_name,
        "level": level,
        "message": message,
        "additional_data": additional_data,
        "time": time,
        "api_name": api_name
    }
    log_collection.insert_one(log_entry)

def log_device_health():
    # Get CPU usage
    cpu_usage = psutil.cpu_percent()

    # Get memory usage
    memory_usage = psutil.virtual_memory().percent
    #logging.info(f"Memory Usage: {memory_usage}%")

    # Get GPU information  (assuming you have a GPU and it's supported by psutil)
    gpu_info = []
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_info.append({"name": gpu.name, "usage": gpu.load * 100})
            #logging.info(f"GPU {i + 1} - Name: {gpu.name}, GPU Usage: {gpu.load * 100}%")
    except ImportError:
        logging.warning("GPUtil module not found. GPU information not logged.")

    # Get other system information
    current_working_directory = os.getcwd()
    #logging.info(f"Current Working Directory: {current_working_directory}")
    system_information = platform.uname()
    #logging.info(f"System Information: {system_information}")

    # Return gathered information
    health_details = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_info": gpu_info,
        "current_working_directory": current_working_directory,
        "system_information": system_information
    }
    return health_details


#convert Base 64 to PDF
def base64_to_pdf(base64_data, output_path):
    # Decode the base64 string
    pdf_data = base64.b64decode(base64_data)
    # Write the decoded data to a PDF file
    with open(output_path, 'wb') as pdf_file:
        pdf_file.write(pdf_data)
    
def generate_pdf_path(user_id, file_name, base_folder="Document"):
    # Create a folder based on the user ID
    user_folder = os.path.join(base_folder, f"{user_id}")
    os.makedirs(user_folder, exist_ok=True)

    # Dynamic output path for the PDF file
    output_path = os.path.join(user_folder, file_name)

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
            text_path=os.path.join(pdf_folder_path, file)
            loader = TextLoader(text_path, encoding='utf-8')
            pdf_documents.extend(loader.load())
    if len(pdf_documents) == 0:
        return None
    return pdf_documents
    

# Load Faiss vector database
def load_faiss_vector_database(docs, embeddings, db_path):
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_path)
    return db

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# loading main db
main_db = FAISS.load_local(main_db_directory, embeddings)

# Load Faiss vector database
def load_main_db():
    return FAISS.load_local(main_db_directory, embeddings)

# Background task to reload the vector database
def reload_vector_database():
    global main_db
    main_db = load_main_db()

##### 
def manage_system_prompt(template):
    B_INST, E_INST = "<s>[INST]", "\n[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n"
    
    SYSTEM_PROMPT = B_SYS + template + E_SYS
    context = '{context}\n'
    qa_query = 'query: {query}'
    template =  B_INST + SYSTEM_PROMPT + context + qa_query + E_INST
    return template

# Callback manager for LLM
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LLM configuration
llm = LlamaCpp(
        model_path=model_path,  # Ensure model_path is defined
        max_tokens=max_token,
        temperature=temperature,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        n_ctx=n_ctx,
        verbose=verbose
)


prompt = PromptTemplate(
input_variables=["query","context"], template=manage_system_prompt(template)
)

chain = load_qa_chain(
    llm, chain_type="stuff",prompt=prompt
)

@app.post('/Ron')
async def search_results(request: SearchInput, background_tasks: BackgroundTasks):
    startTime = datetime.now()
    health_details = log_device_health()
    try:
        logging.info("Entry from search_results Method")
        logging.info(f"User Name: {request.user_name}")
        log_to_mongo(user_name=request.user_name,level="info", message="Entry from Ron_search_results Method", time=startTime,api_name="Ron",additional_data=health_details)
    
        # Reload the vector database in the background
        reload_vector_database()
        # Perform search and get response
        query = request.query.lower()
        logging.info(f"Question: {query}")
        log_to_mongo(user_name=request.user_name,level="info", message=f"Question: {query}",time=startTime,api_name="Ron",additional_data=health_details)
        logging.info(f"similarity_search_entry: {query}")
        docs=main_db.similarity_search(query)
        logging.info(f"similarity_search_exit: {docs}")
        #chat_history = chain.memory.buffer or ""  # Corrected method name
        logging.info("lama_chain_entry:")
        response = chain({"input_documents": docs, "query": query}, return_only_outputs=True)
        logging.info("lama_chain_exit_response_genrated:")
        source_info = []
        if query not in greet:
            for result in docs:
                file_path = result.metadata['source']
                file_path_parts = file_path.split("/")
                pdf_name = file_path_parts[-1]
                # Create a web URL by joining the base URL and PDF file name
                base_url = actual_pdf_base_url
                web_url = urljoin(base_url, pdf_name)
                source_info.append({'link': web_url, 'name': pdf_name})
        unique_links = set()
        unique_source_info = []
        for item in source_info:
            link = item["link"]
            if link not in unique_links:
                unique_links.add(link)
                unique_source_info.append(item)
        logging.info(f"Response:{response}, source_info:{unique_source_info}")

        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        log_to_mongo(user_name=request.user_name, level="info", message=f"Response:{response}, source_info:{unique_source_info}", time=elapsed_time.total_seconds(), api_name="Ron", additional_data=health_details)

        logging.info(f"Exit from search_results Method at {elapsed_time.total_seconds() * 1000} ms")
        log_to_mongo(user_name=request.user_name, level="info", message="Exit from RON search_results Method", time=elapsed_time.total_seconds(), api_name="Ron", additional_data=health_details)

        return {"response": response, "source_info": unique_source_info}
    
    except Exception as e:
        log_to_mongo(user_name=request.user_name, level="error", message=f"An error occurred while search_results response. Details : {str(e)}", time=startTime, api_name="Ron")
        logging.error(f"An error occurred while search_results response. Details : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Function to handle PDF processing and database creation
def process_pdf_and_create_db(request):
    startTime = datetime.now()
    health_details = log_device_health()
    elapsed_time = None
    try:
        user_name = request.user_name
        file_name = request.file_name
        user_temp_folder = f'/home/sachin/RON/RON_APP/Temp/vectorstore/{user_name}'
        file_path = generate_pdf_path(user_name,file_name)
        #print(file_path)
        log_to_mongo(user_name=request.user_name,level="info", message="PDF processing and database creation function call", time=startTime,api_name="process_pdf_and_create_db_function",additional_data=health_details)
        logging.info(f"User Name: {request.user_name}")
        #print(request.file_data)
        if request.file_data:
            base64_to_pdf(request.file_data, file_path)
            file_dir = f"{file_path.split('/')[0]}/{file_path.split('/')[1]}"
            #print("file path",file_dir)
            pdf_documents = load_pdf_documents(file_dir)
            #print(pdf_documents)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(pdf_documents)

            # Create user-specific temp folder if not exists
            if not os.path.exists(user_temp_folder):
                os.makedirs(user_temp_folder)

            Temp_db = load_faiss_vector_database(docs, embeddings, f'{user_temp_folder}/db_faiss')
            logging.info("File uploaded successfully")
            
            log_to_mongo(user_name=request.user_name,level="info", message="File uploaded/Proceesed successfully and temp vector DB created", time=startTime,api_name="process_pdf_and_create_db_function",additional_data=health_details)
    
            #source_folder = 'Document/viren.vaishnav@rysun.com'
            source_folder = file_dir
            base_destination_folder = "/home/sachin/RON/RON_APP/Document/Processed_Data/"
            destination_folder = f"{base_destination_folder}{source_folder.split('/')[1]}"
            log_to_mongo(user_name=request.user_name,level="info", message=f"processed File moved successfully to {destination_folder}: {base_destination_folder}", time=elapsed_time,api_name="process_pdf_and_create_db_function",additional_data=health_details)
    
            counter = 1
            while os.path.exists(destination_folder):
                destination_folder = f"{base_destination_folder}{source_folder.split('/')[1]}_{counter}"
                counter += 1

            shutil.move(source_folder, destination_folder)
        else:
            # Check if the database already exists
            db_path = f'{user_temp_folder}/db_faiss'
            if os.path.exists(db_path):
                Temp_db = FAISS.load_local(db_path, embeddings)
                logging.info("Loaded existing database")
                log_to_mongo(user_name=request.user_name,level="info", message="Loaded existing database", time=elapsed_time,api_name="process_pdf_and_create_db_function",additional_data=health_details)
    
            else:
                logging.info("No new PDF or existing database found")
                log_to_mongo(user_name=request.user_name,level="info", message="No new PDF or existing database found", time=elapsed_time,api_name="process_pdf_and_create_db_function",additional_data=health_details)
                return None  # No PDF or database, return None

        return Temp_db,docs

    except Exception as e:
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        log_to_mongo(user_name=request.user_name,level="error", message=f"Error processing PDF and creating database: {str(e)}", time=elapsed_time,api_name="process_pdf_and_create_db_function",additional_data=health_details)
        logging.error(f"Error processing PDF and creating database: {str(e)}")
        return None


#################################################################################################

@app.post('/Document_Training')
async def document_training(request: DocInput):
    startTime = datetime.now()
    health_details = log_device_health()
    elapsed_time = None
    try:
        log_to_mongo(user_name=request.user_name,level="info", message="Entry from Document_Training Method", time=startTime,api_name="Document_Training_Q&A",additional_data=health_details)
        logging.info("Entry from Document_Training Method")
        logging.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        query = request.query.lower()
        log_to_mongo(user_name=request.user_name,level="info", message=f"Question: {query}", time=startTime,api_name="Document_Training_Q&A",additional_data=health_details)
        logging.info(f"Question: {query}")
        prompt = PromptTemplate(input_variables = ["query","context"], template=manage_system_prompt(prompt_Document_Training))
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(query)
        response = llmChain({"input_documents": docs, "query": query}, return_only_outputs=True)
        logging.info(f"Response:{response}")
        
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="info", message=f"Document_Training_Q&A_Response:{response}", time=elapsed_seconds,api_name="Document_Training_Q&A",additional_data=health_details)
 
        log_to_mongo(user_name=request.user_name,level="info", message="Exit from Document_Training Method", time=elapsed_seconds,api_name="Document_Training_Q&A",additional_data=health_details)
        logging.info("Exit from Document_Training Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        return {"response": response}
    
    except Exception as e:
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="error", message=f"Error in Document_Training method: {str(e)}", time=elapsed_seconds,api_name="Document_Training_Q&A",additional_data=health_details)
        logging.error(f"Error in Document_Training method: {str(e)}")
        return {"response": f"Error in Document_Training method: {str(e)}"}


#####################################################################################################
    
@app.post('/Summarization')
async def Summarization(request: DocInput):
    startTime = datetime.now()
    health_details = log_device_health()
    elapsed_time = None
    try:
        log_to_mongo(user_name=request.user_name,level="info", message="Entry from Summarization Method", time=startTime, api_name="Summarization",additional_data=health_details)
        logging.info("Entry from Summarization Method")
        logging.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        query = "What are the main takeaways from the document, and can you provide a brief summary of its key points?"
        logging.info(f"Question: {query}")
        prompt = PromptTemplate(input_variables = ["query", "context"], template=manage_system_prompt(promptTemplateTextSummerization))
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(query)
        response = llmChain({"input_documents": docs, "query": query}, return_only_outputs=True)
        logging.info(f"Response:{response}")
        
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="info", message=f"Summarization_Response:{response}", time=elapsed_seconds,api_name="Summarization",additional_data=health_details)
        
        logging.info("Exit from Summarization Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        log_to_mongo(user_name=request.user_name,level="info", message="Exit from Summarization Method", time=elapsed_seconds,api_name="Summarization",additional_data=health_details)
        return {"response": response}

    except Exception as e:
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="error", message=f"Error in Summarization method: {str(e)}", time=elapsed_seconds,api_name="Summarization",additional_data=health_details)
        logging.error(f"Error in Summarization method: {str(e)}")
        return {"response": f"Error in Summarization method: {str(e)}"}
    
#######################################################################################################
@app.post('/NER')
async def ner(request: DocInput):
    startTime = datetime.now()
    health_details = log_device_health()
    elapsed_time = None
    try:
        log_to_mongo(user_name=request.user_name,level="info", message="entry from ner api", time=startTime,api_name="NER",additional_data=health_details)
        logging.info("Entry from ner Method")
        logging.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)
        #print(docs)
        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        query = "extract All the entity from this document"
        logging.info(f"Question: {query}")
        prompt = PromptTemplate(input_variables = ["query", "context"], template=manage_system_prompt(NER_prompt))
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(query)
        response = llmChain({"input_documents": docs, "query": query}, return_only_outputs=True)
        logging.info(f"Response:{response}")
        
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime

        # Convert timedelta to total seconds
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="info", message=f"NER_Response:{response}", time=elapsed_seconds,api_name="NER",additional_data=health_details)

        logging.info("Exit from ner Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        log_to_mongo(user_name=request.user_name,level="info", message="Exit from ner Method", time=elapsed_seconds,api_name="NER",additional_data=health_details)

        return {"response": response}

    except Exception as e:
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="error", message=f"Error in ner method: {str(e)}", time=elapsed_seconds,api_name="NER",additional_data=health_details)
        logging.error(f"Error in ner method: {str(e)}")
        return {"response": f"Error in ner method: {str(e)}"}
        
        
##################################################################################################

@app.post('/Classification')
async def classification(request: DocInput):
    startTime = datetime.now()
    health_details = log_device_health()
    elapsed_time = None
    try:
        log_to_mongo(user_name=request.user_name,level="info", message="Entry from classification Method", time=startTime,api_name="Classification",additional_data=health_details)
        logging.info("Entry from classification Method")
        logging.info(f"User Name: {request.user_name}")
        # Process PDF and create database
        Temp_db,docs = process_pdf_and_create_db(request)

        # If no new PDF or existing database found, return a response
        if Temp_db is None:
            return {"response": "Error processing PDF or creating database"}

        query = "classify this document for me and give me only multiple classify catagory not additinal information"
        logging.info(f"Question: {query}")
        prompt = PromptTemplate(input_variables = ["query", "context"], template=manage_system_prompt(promptTemplateClassification))
        llmChain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        docs = Temp_db.similarity_search(query)
        response = llmChain({"input_documents": docs, "query": query}, return_only_outputs=True)
        logging.info(f"Response:{response}")
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime

        # Convert timedelta to total seconds
        elapsed_seconds = elapsed_time.total_seconds()
        log_to_mongo(user_name=request.user_name,level="info", message=f"Classification_Response:{response}", time=elapsed_seconds,api_name="Classification",additional_data=health_details)
        logging.info("Exit from classification Method at " + str((stopTime - startTime) * (10 ** 3)) + " ms")
        log_to_mongo(user_name=request.user_name,level="info", message="Exit from classification Method", time=elapsed_seconds,api_name="Classification",additional_data=health_details)
        return {"response": response}

    except Exception as e:
        stopTime = datetime.now()
        elapsed_time = stopTime - startTime
        elapsed_seconds = elapsed_time.total_seconds()
        logging.error(f"Error in classification method: {str(e)}")
        log_to_mongo(user_name=request.user_name,level="info", message=f"Error in classification method: {str(e)}", time=elapsed_seconds,api_name="Classification",additional_data=health_details)
        return {"response": "Internal server error"}

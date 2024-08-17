import os
from PyPDF2 import PdfReader
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.llms import Ollama, HuggingFaceHub
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_community.vectorstores import Qdrant, FAISS
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pdfplumber
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

huggingface_hub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
app = Flask(__name__, template_folder='templates3')
app.secret_key = 'your_secret_key_here'  # Required for session
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

qa_chain = None
uploaded_file = None
ats_score = None
chat_history = []
model = None
embeddings = None
mode = None  # Keep track of the selected mode

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def setup_qa_system(selected_mode):
    global model, embeddings, mode
    mode = selected_mode  # Set the mode
    
    if mode == 'offline':
        model = Ollama(model="gemma2:2b")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")  # This is a fast embedding model
    else:
        model = HuggingFaceHub(repo_id="google/gemma-2-2b-it", huggingfacehub_api_token=huggingface_hub_api_token)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # This is a small, fast model

    # Test embeddings
    test_embedding = embeddings.embed_query("Test sentence")
    logging.debug(f"Embedding dimension: {len(test_embedding)}")

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def setup_qa_chain(file_path):
    parser = StrOutputParser()
    
    # Extract text from PDF using extract_text_from_pdf function
    text = extract_text_from_pdf(file_path)
    
    # Split the text into chunks
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # Create document objects
    from langchain.docstore.document import Document
    documents = [Document(page_content=chunk) for chunk in chunks]

    template = """
    Answer the question based on the context below. If you cannot answer the question, 
    reply "I don't know the answer."

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)

    # Create vectorstore based on mode
    if mode == 'offline':
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Adjust k as needed

    logging.debug(f"Retriever setup complete: {retriever}")
    return (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | prompt
        | model
        | parser
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_choice = request.form['model_choice']
        session['model_choice'] = model_choice
        setup_qa_system(model_choice)
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global uploaded_file, qa_chain
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_file = filename
            
            qa_chain = setup_qa_chain(file_path)
            
            return redirect(url_for('ask_question'))
    return render_template('upload.html')

@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    global qa_chain, uploaded_file
    answer = ""
    if request.method == 'POST':
        question = request.form['question']
        if qa_chain:
            try:
                logging.debug(f"Received question: {question}")
                response = qa_chain.invoke({"question": question})
                
                answer_parts = response.split(']')
                if len(answer_parts) > 1:
                    raw_answer = answer_parts[-1].strip()
                    answer = raw_answer.strip("' ")
                    if answer.count('\n') > 1:
                        answer = answer.split('\n')
                        answer = [point.strip() for point in answer if point.strip()]
                else:
                    answer = response
                logging.debug(f"Generated answer: {answer}")
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                answer = f"An error occurred: {str(e)}"
    return render_template('ask.html', answer=answer, uploaded_file=uploaded_file)

def extract_text_from_resume(resume_file):
    reader = PdfReader(resume_file)
    number_of_pages = len(reader.pages)
    text = ''
    for page in range(number_of_pages):
        page_text = reader.pages[page].extract_text()
        text += page_text
    return text

def calculate_ats_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(np.clip(similarity * 100, 0, 100), 2)  # Clip score to range [0, 100]

@app.route('/ats', methods=['GET', 'POST'])
def calculate_ats():
    if request.method == 'POST':
        resume_file = request.files['resume']
        resume_text = extract_text_from_resume(resume_file)
        job_description = request.form['job_description']
        ats_score = calculate_ats_score(resume_text, job_description)
        return render_template('ats.html', ats_score=ats_score)
    return render_template('ats.html')


from hugchat import hugchat
from hugchat.login import Login
# Load environment variables from .env file
load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWD= os.getenv("PASSWD")

cookie_path_dir = "./cookies/" # NOTE: trailing slash (/) is required to avoid errors
sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

# Create your ChatBot
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

# message_result = chatbot.chat("Hi!") 










@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html', chat_history=chat_history)

@app.route('/api/chat', methods=['POST'])

def api_chat():
    global chat_history, model


    if mode == 'offline':
        
        user_message = request.json['message']
        # chat_history.append(("User", user_message))
        
        response = model.invoke(user_message)
        # chat_history.append(("AI", response))
        
        return jsonify({
            "user_message": user_message,
            "ai_response": response
        })
    else:
       

        user_message = request.json['message']

        response = chatbot.chat(user_message)
        response_text = response.text  # Extract the text from the Message object

        # chat_history.append(("AI", response_text))

        return jsonify({
            "user_message": user_message,
            "ai_response": response_text
        })

if __name__ == '__main__':
    app.run(debug=True)

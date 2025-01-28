import os
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from urllib.parse import urljoin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

FAISS_DB_DIR = 'faiss_index_internal'
MAX_RETRIES = 3
RETRY_DELAY = 5

def scrape_website(url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
            return text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to retrieve {url}: {e}")
            time.sleep(RETRY_DELAY)
    print(f"Failed to retrieve {url} after {MAX_RETRIES} attempts.")
    return ""

def get_absolute_links(url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            links = [link for link in links if not link.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            return list(set(links))
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to retrieve links from {url}: {e}")
            time.sleep(RETRY_DELAY)
    print(f"Failed to retrieve links from {url} after {MAX_RETRIES} attempts.")
    return []

def initialize_qa_chain(urls):
    if os.path.exists(FAISS_DB_DIR):
        print("Loading existing FAISS vectorDB...")
        vectordb = FAISS.load_local(FAISS_DB_DIR, SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2"), allow_dangerous_deserialization=True)
    else:
        print("Creating new embeddings from provided URLs...")
        documents = []
        for url in urls:
            website_text = scrape_website(url)
            if website_text:
                document = Document(page_content=website_text, metadata={"source": url})
                documents.append(document)
            internal_links = get_absolute_links(url)
            for link in internal_links:
                internal_text = scrape_website(link)
                if internal_text:
                    document = Document(page_content=internal_text, metadata={"source": link})
                    documents.append(document)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=90)
        splits = text_splitter.split_documents(documents)
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        vectordb = FAISS.from_documents(splits, embeddings)
        vectordb.save_local(FAISS_DB_DIR)
        print(f"FAISS vectorDB saved at {FAISS_DB_DIR}.")
    CHECKPOINT = "MBZUAI/LaMini-T5-738M"
    TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
    pipe = pipeline('text2text-generation', model=BASE_MODEL, tokenizer=TOKENIZER, max_length=1024, do_sample=True, temperature=0.3, top_p=0.95)
    llm = HuggingFacePipeline(pipeline=pipe)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 10}), return_source_documents=True)
    return qa_chain

def process_answer(instruction, qa_chain):
    result = qa_chain.invoke({"query": instruction})
    source_docs = result.get('source_documents', [])
    answer = result['result'].strip()
    if len(source_docs) == 0 or "does not provide information" in answer.lower() or "couldn't find" in answer.lower() or "not mentioned" in answer.lower():
        return "Sorry, I couldn't find the answer to your question in the provided context."
    else:
        source_url = source_docs[0].metadata.get("source", "Unknown source")
        return f"{answer}\n\nSource: {source_url}"

def get_urls_from_file(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return []
    with open(filename, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

@app.route('/askQuestion', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Invalid request. Please provide a question.'}), 400
        question = data['question']
        filename = 'Internal_links.txt'
        urls = get_urls_from_file(filename)
        if not urls:
            return jsonify({'error': 'No URLs found for processing.'}), 400
        global qa_chain
        if not qa_chain:
            qa_chain = initialize_qa_chain(urls)
        answer = process_answer(question, qa_chain)
        return jsonify({ 'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    qa_chain = None
    app.run(debug=True)
 
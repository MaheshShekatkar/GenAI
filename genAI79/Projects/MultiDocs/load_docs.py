import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

def load_docs():
    documents = []
    for file in os.listdir('docs'):
        if file.endswith('.pdf'):
            pdf_path = ".\docs\\" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('docx') or file.endswith('.doc'):
            doc_path = ".\docs\\" + file            
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('txt'):
            txt_path = ".\txt\\" + file            
            loader = TextLoader(txt_path)
            documents.extend(loader.load())    
            
    return documents            
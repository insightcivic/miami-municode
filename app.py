from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Lazy initialization placeholder
embeddings = None
vectorstore = None
qa = None

def initialize_components():
    global embeddings, vectorstore, qa
    if embeddings is None and vectorstore is None and qa is None:
        # Load and preprocess the text file
        loader = TextLoader("code.txt")
        documents = loader.load()

        # Efficient text splitting to ensure small manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Create the vectorstore with a focus on managing chunk size
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Initialize the language model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")

        # Initialize the retrieval chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

@app.route('/', methods=['GET', 'POST'])
def index():
    initialize_components()  # Ensure components are initialized before use
    if request.method == 'POST':
        question = request.form['question']
        max_length = 300
        if len(question) > max_length:
            question = question[:max_length] + '...'
        try:
            response = qa.run(question)
        except Exception as e:
            app.logger.error(f"Error during question handling: {str(e)}")
            response = f"Sorry, there was an error processing your question. Please try again later."
        return render_template('index.html', response=response, question=question)
    return render_template('index.html', response=None, question=None)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
import nltk
from flask import flash, redirect, session, url_for
import ssl
import openai


app = Flask(__name__)

class Chatbot:
    def __init__(self, directory_path, openai_api_key):
        self.directory_path = directory_path
        self.openai_api_key = openai_api_key
        self.loader = DirectoryLoader(self.directory_path, glob='**/*.txt')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = self.text_splitter.split_documents(self.documents)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.docsearch = Chroma.from_documents(self.texts, self.embeddings)
        self.qa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0.5), chain_type="stuff", vectorstore=self.docsearch)
        nltk.download('punkt')

    def ask_question(self, question):
        return self.qa.run(question)

#env variable
OPENAIKEY='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'
databox='./sample_txt/fmvss/'

bot = Chatbot(databox, OPENAIKEY)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get_bot_response", methods=["POST"])
def get_bot_response():
    input_question = request.form["question"]
    answer = bot.ask_question(input_question)
    return render_template("home.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

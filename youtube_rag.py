from rich.console import Console
from halo import Halo
import asyncio
import os
import openai
from operator import itemgetter
import glob
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# Load the environment variables from the .env file
load_dotenv()

# Set the OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Get a list of all .txt files in the 'yt-summaries-data' directory
txt_files = glob.glob('yt-summaries-data/*.txt')

class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

# Load each file into the raw_documents list
raw_documents = []
for file in txt_files:
    try:
        with open(file, 'r') as f:
            text = f.read()
            raw_documents.append(Document(text))
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        continue

# Load existing vector database if it exists, otherwise create a new one
vector_db_path = 'yt-summaries-data/vector_db.faiss'
if os.path.exists(vector_db_path):
    db = FAISS.load_local(vector_db_path, OpenAIEmbeddings())
else:
    db = FAISS.from_documents(raw_documents, OpenAIEmbeddings())

model = ChatOpenAI()

# Create a retriever from the vector store
retriever = db.as_retriever()

# TEST DOC RETRIEVAL
# query = "What is next.js?"
# try:
#     docs = retriever.get_relevant_documents(query)
#     print(docs[0].page_content)
# except Exception as e:
#     print(f"Error performing similarity search: {e}")

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# print(chain.invoke({"question": "what is next.js?", "language": "english"}))

# load cli libs
console = Console()
spinner = Halo(text='AI is thinking', spinner='dots')

async def conversation_loop():
    response_buffer = ""  # Buffer to store the AI's response

    while True:
        user_input = console.input("[bold cyan]User[/bold cyan]: ")  # Get user input from the command-line
        if user_input.lower() == "exit":
            break  # Exit the conversation loop if the user enters "exit"

        spinner.start()
        async for output in chain.astream({"question": user_input, "language": "english"}):
            if isinstance(output, str):
                response_buffer += output
            else:
                response_buffer += output["content"]

            # Check if the response buffer contains a complete response
            if response_buffer.endswith("."):
                spinner.stop()
                console.print("[bold green]AI[/bold green]:", response_buffer.strip())  # Print the complete response
                response_buffer = ""  # Clear the response buffer  # Clear the response buffer  # Clear the response buffer

if __name__ == "__main__":
    asyncio.run(conversation_loop())
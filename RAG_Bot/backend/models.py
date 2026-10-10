from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

#   Embedding Model:
model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


#   Chat Model:


llm= ChatGroq(
    model='openai/gpt-oss-20b'
)

#   BACKUP MODELS:
#1. gemini-3.1-flash-lite
#2. gemini-3.5-flash
#3. openai/gpt-oss-20b
#4. qwen/qwen3.8-27b
from dotenv import load_dotenv

# LLMs
from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama

# loaders
from langchain_community.document_loaders import WebBaseLoader

# embeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

# vector stores
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

SOURCE = "https://docs.smith.langchain.com/overview/"
QUESTION = "how can langsmith help with testing?"

# SOURCE = "https://news.samsung.com/global/enter-the-new-era-of-mobile-ai-with-samsung-galaxy-s24-series"
# QUESTION = "what were the features that were new to the new Samsung Galaxy S24?"

print("Loading LLM...", end="")
llm = ChatOpenAI()
# llm = Ollama(model='llama2')
print(" Done.")

print("Loading documents...", end="")
loader = WebBaseLoader(SOURCE)
docs = loader.load()
print(" Done.")

print("Splitting documents...", end="")
textSplitter = RecursiveCharacterTextSplitter()
documents = textSplitter.split_documents(docs)
print(" Done.")

print("Creating vector store...", end="")
embeddings = OpenAIEmbeddings()
# embeddings = OllamaEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
print(" Done.")

retriever = vector.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chatHistory"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retrieverChain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chatHistory"),
    ("user", "{input}"),
])
documentChain = create_stuff_documents_chain(llm, prompt)

retrievalChain = create_retrieval_chain(retrieverChain, documentChain)

chatHistory = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# chatHistory = [HumanMessage(content="Were there any groundbreaking features to the new Samsung Galaxy S24?"), AIMessage(content="Yes!")]

print("Retrieving answer...", end="")
response = retrievalChain.invoke({
    "chatHistory": chatHistory,
    "input": "Tell me how"
    # "input": "What were they?"
})
print(" Done.")

print(response)

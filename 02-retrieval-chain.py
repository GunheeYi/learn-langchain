from dotenv import load_dotenv

# LLMs
# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# loaders
from langchain_community.document_loaders import WebBaseLoader

# embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# vector stores
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# SOURCE = "https://docs.smith.langchain.com/overview/"
# QUESTION = "how can langsmith help with testing?"

SOURCE = "https://news.samsung.com/global/enter-the-new-era-of-mobile-ai-with-samsung-galaxy-s24-series"
QUESTION = "what were the features that were new to the new Samsung Galaxy S24?"

llm = ChatOpenAI()
# llm = Ollama(model='llama2')

loader = WebBaseLoader(SOURCE)

docs = loader.load()

embeddings = OpenAIEmbeddings()
# embeddings = OllamaEmbeddings()

textSplitter = RecursiveCharacterTextSplitter()

documents = textSplitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
    {context}
</context>

Question: {input}""")

documentChain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrievalChain = create_retrieval_chain(retriever, documentChain)

response = retrievalChain.invoke({ "input": QUESTION })
print(response['answer'])


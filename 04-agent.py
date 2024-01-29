from dotenv import load_dotenv

# LLMs
from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama

# loaders
from langchain_community.document_loaders import WebBaseLoader

# embeddings
from langchain_openai import OpenAIEmbeddings

# vector stores
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.tools.retriever import create_retriever_tool

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor

load_dotenv()

SOURCE = "https://docs.smith.langchain.com/overview/"
QUESTION = "how can langsmith help with testing?"

# SOURCE = "https://news.samsung.com/global/enter-the-new-era-of-mobile-ai-with-samsung-galaxy-s24-series"
# QUESTION = "what were the features that were new to the new Samsung Galaxy S24?"

print("Loading LLM...", end="")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
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

retrieverTool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
)
search = TavilySearchResults()
tools = [retrieverTool, search]

prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt)

agent = create_openai_functions_agent(llm, tools, prompt)
agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agentExecutor.invoke({ "input": "how can langsmith help with testing?" })

# agentExecutor.invoke({ "input": "what is the weather in Busan?" })

chatHistory = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
agentExecutor.invoke({ "chat_history": chatHistory, "input": "Tell me how" })

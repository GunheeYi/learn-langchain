import os
from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# llm = ChatOpenAI()
llm = Ollama(model='llama2')

outputParser = StrOutputParser()

chain = prompt | llm | outputParser

print(chain.invoke({ "input": "how can langsmith help with testing?" }))

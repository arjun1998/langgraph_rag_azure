from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.tools import tool
load_dotenv()
persist_directory="./vector_db"
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(collection_name = "hybrid_collection",
                    embedding_function = embeddings,
                    persist_directory=persist_directory)

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve information to help answer a query."""    
    retrieved_docs = vector_store.similarity_search(query,k=5)
    serialized ="\n\n".join(
        (f"Source:{doc.metadata}\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized,retrieved_docs

from langchain.agents import create_agent

tools = [retrieve_context]

prompt =(
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
model = ChatOpenAI(model="gpt-4.1")
agent = create_agent(model, tools, system_prompt=prompt)

query=(
    ##"What is the standard method for Task Decomposition?\n\n"
    ##"Once you get the answer, look up common extensions of that method."
    "Tell me the details of all the hyundai creta"
)

for event in agent.stream(
    {"messages":[{"role":"user","content":query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
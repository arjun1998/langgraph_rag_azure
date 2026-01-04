from nodes.ingest import GraphState
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
def retrieve_node(state:GraphState):
    query=state["query"]

    persist_directory="./vector_db"
    azure_endpoint="https://azure-openai-wk.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
    embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large",
                                    azure_endpoint=azure_endpoint,
                                    api_key=os.getenv("AZURE_OPENAI_API_KEY"))

    vector_store = Chroma(collection_name = "hybrid_collection",
                        embedding_function = embeddings,
                        persist_directory=persist_directory)

     
    retrieved_docs = vector_store.similarity_search(query,k=5)
    return {"results":[doc.page_content for doc in retrieved_docs]}

    
from nodes.ingest import GraphState
from dotenv import load_dotenv
import os

from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI

import time
load_dotenv()

model = AzureChatOpenAI(azure_deployment="gpt-4o",
                        api_version="2024-12-01-preview",
                           temperature=0,
                           api_key=os.getenv("AZURE_OPENAI_INFERENCE_API_KEY"),
                           azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                           )


def answer_node(state:GraphState):
    query=state["query"]


    context = "\n".join(state["results"])
    response = model.invoke(f"Answer the query:{query}\n\n Context:{context} ")
    return {"results":[response.content]}


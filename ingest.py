
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import bs4
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


embeddings=OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = Chroma(collection_name = "hybrid_collection",
                      embedding_function = embeddings,
                      persist_directory="./vector_db")


loader= CSVLoader(file_path="data/train.csv")

csv_data = loader.load()

bs4_strainer=bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader =WebBaseLoader(web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs={"parse_only":bs4_strainer}, )

docs=loader.load()

total_docs=docs+csv_data[:100]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits=text_splitter.split_documents(total_docs)

print(f"Split the docs into {len(all_splits)} sub-documents")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
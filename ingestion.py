from dotenv import load_dotenv
from langchain_tavily import TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

POSTS_URL = "https://lilianweng.github.io/posts"

load_dotenv()

tavily_map = TavilyMap(
    max_depth=5,
    max_breadth=30,
    max_pages=200,
    select_paths=[r"/posts/\d{4}-\d{2}-\d{2}-.*"],
)

map_result = tavily_map.invoke({"url": POSTS_URL})
urls = map_result["results"]

docs = [
    UnstructuredLoader(
        web_url=url, chunking_strategy="basic", max_characters=1000000
    ).load()
    for url in urls
]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
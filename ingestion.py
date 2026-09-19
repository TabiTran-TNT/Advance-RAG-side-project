from dotenv import load_dotenv
from langchain_tavily import TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

POSTS_URL = "https://lilianweng.github.io/posts"
COLLECTION = "rag-chroma"
PERSIST_DIR = "./.chroma"


def ingest() -> None:
    tavily_map = TavilyMap(
        max_depth=5,
        max_breadth=30,
        max_pages=200,
        select_paths=[r"/posts/\d{4}-\d{2}-\d{2}-.*"],
    )
    urls = tavily_map.invoke({"url": POSTS_URL})["results"]

    docs = [
        UnstructuredLoader(
            web_url=url, chunking_strategy="basic", max_characters=1000000
        ).load()
        for url in urls
    ]
    docs_list = [item for sub in docs for item in sub]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = splitter.split_documents(docs_list)

    Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR,
    )


# Only open existing Chroma DB
retriever = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()


if __name__ == "__main__":
    ingest()
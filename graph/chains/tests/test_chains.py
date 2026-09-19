from dotenv import load_dotenv

load_dotenv()

from graph.chains.retriever_grade import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain

from ingestion import retriever

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"

def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make a paper", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"

    docs = retriever.invoke(question)

    generation = generation_chain.invoke({
        "context": [doc.page_content for doc in docs],
        "question": question,
    })

    assert generation is not None

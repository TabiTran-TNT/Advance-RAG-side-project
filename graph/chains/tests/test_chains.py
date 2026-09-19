from dotenv import load_dotenv

load_dotenv()

from graph.chains.retriever_grade import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
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

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    docs_txt = [doc.page_content for doc in docs]

    generation = generation_chain.invoke({
        "context": docs_txt,
        "question": question,
    })
    
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs_txt, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": [doc.page_content for doc in docs],
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score

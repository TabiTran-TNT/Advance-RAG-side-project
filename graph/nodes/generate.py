from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]
    
    generation = generation_chain.invoke({
        "context": [doc.page_content for doc in documents],
        "question": question,
    })
    return {"documents": documents, "question": question, "generation": generation}

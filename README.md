# Side Project about Agentic RAG with LangGraph 


Implementation of Reflective RAG, Self-RAG & Adaptive RAG tailored towards developers and production-oriented applications for my learning LangGraph.

This repository contains a refactored version of the original [LangChain's Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain).

Through this project, I gained a solid understanding of how LangGraph and LangChain work together within a RAG framework. I also implemented comprehensive test coverage to ensure the reliability, stability, and correctness of the application, enabling developers to confidently validate their implementations. The codebase was designed with production readiness in mind, making it easier to transition from experimentation and development to deployment.


## What I have learned

* **Agentic RAG Implementation**: Develop an intelligent RAG system capable of dynamically deciding when and what information to retrieve based on the user’s query.
* **Graph-Based Control Flow**: Leverage LangGraph to design and manage sophisticated, stateful workflows throughout the RAG pipeline.
* **Document Relevance Evaluation**: Implement document grading mechanisms to assess retrieval quality and identify potential hallucinations in generated responses.
* **Adaptive Information Retrieval**: Build a flexible retrieval strategy that can dynamically switch between local knowledge sources and web search when additional information is required.
* **State Management**: Implement robust state management to maintain context and coordinate complex information flows across multiple stages of the application.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # For web search capabilities
LANGSMITH_API_KEY=your_langchain_api_key_here  # Optional, for tracing
LANGSMITH_TRACING=true                      # Optional
LANGCHAIN_PROJECT=agentic-rag                  # Optional
LANGSMITH_ENDPOINT=                   # Optional if you are not in US Region
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/emarco177/langgraph-course.git
cd langgraph-course
```
## Install Dependencies

### Using `uv`

If the project already has a `pyproject.toml`:

```bash
uv sync
```

Then activate the virtual environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

### Using `pip`

Create and activate a virtual environment:

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Clone the project:

```bash
git clone https://github.com/TabiTran-TNT/Advance-RAG-side-project.git
```

Go to the project directory:

```bash
cd Advance-RAG-side-project
```

### Start the Agentic RAG flow

Using `uv`:

```bash
uv run python main.py
```

Or, if the virtual environment has been activated:

```bash
python main.py
```

## Running Tests

Using `uv`:

```bash
uv run pytest . -s -v
```

Or, if the virtual environment has been activated:

```bash
pytest . -s -v
```

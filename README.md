# TranslationLLMapp_Langchain:

An application leveraging LangChain and OpenAI's GPT models to facilitate multilingual document translation and retrieval-augmented generation (RAG).

## Project Overview

This project integrates LangChain with OpenAI's language models to enable efficient translation of documents and retrieval-augmented generation tasks. It utilizes the `rag.py` script for implementing RAG functionalities and processes documents like `The future of space technology.pdf` to demonstrate its capabilities.

## Setup Instructions

### 1. Clone the Repository

Begin by cloning the repository to your local machine:

```bash
git clone https://github.com/Dhwanitank/TranslationLLMapp_Langchain.git
cd TranslationLLMapp_Langchain

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Set Up Environment Variables

OPENAI_API_KEY=your_openai_api_key

### 4. Run

To run the RAG functionality on a document, execute the following command:

python rag.py


### Example:

    Document Input:
    Place your PDF document in the project directory.

    Processing:
    Run rag.py to initiate the translation and RAG processes.

    Output:
    The script will output the translated content and any generated responses based on the document's content.

import os
from transformers import pipeline
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import tempfile
import docx
from PyPDF2 import PdfReader

# --- Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")  # Load API key from .env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")  # Optional for ADC

# --- Step 1: Extract and Split Document Content ---
def load_and_split_document(uploaded_file):
    """Load and split document into manageable chunks."""
    text = ""
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "txt":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
            temp_txt.write(uploaded_file.read())
            temp_txt.flush()
            with open(temp_txt.name, "r", encoding="utf-8") as f:
                text = f.read()
    elif file_extension == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    elif file_extension == "docx":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        st.error("Unsupported file format!")
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# --- Step 2: Create Embeddings and Store in FAISS ---
def create_and_store_embeddings(chunks):
    """Create embeddings and store them in FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# --- Step 3: Set up the QA Chain ---
def get_retrieval_qa_chain(vector_store):
    """Set up RetrievalQA using Google Generative AI."""
    retriever = vector_store.as_retriever()

    # Use Google Generative AI (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)

    # Prompt Template
    prompt_template = """
    Use the following context to answer the question.
    If the answer is not in the context, say 'I don't know.'

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Combine retriever and model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return qa_chain

# --- Step 4: Translate Text ---
def translate_text(output_text, target_language):
    """Translate the generated output text to the user-selected language."""
    try:
        if target_language == "English":
            return output_text  # No translation needed
        # Translation pipelines for different languages
        model_map = {
            "French": "Helsinki-NLP/opus-mt-en-fr",
            "Spanish": "Helsinki-NLP/opus-mt-en-es",
            "German": "Helsinki-NLP/opus-mt-en-de"
        }
        translation_model = model_map.get(target_language)
        if translation_model:
            translator = pipeline("translation", model=translation_model)
            translated_text = translator(output_text, max_length=512)[0]['translation_text']
            return translated_text
        else:
            return "Translation for the selected language is not supported."
    except Exception as e:
        return f"Translation error: {str(e)}"

# --- Step 5: Streamlit UI ---
def main():
    st.title("RAG-based Document Q&A System with Translation")
    st.write("Upload a document (PDF, DOCX, or TXT) and ask questions based on its content. Translated answers are supported!")

    # File upload
    uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}")
        with st.spinner("Processing document..."):
            chunks = load_and_split_document(uploaded_file)
            if chunks:
                vector_store = create_and_store_embeddings(chunks)
                st.success("Document processed and indexed successfully!")

                # QA Chain
                qa_chain = get_retrieval_qa_chain(vector_store)

                # User Question
                user_question = st.text_input("Ask a question about the document:")
                target_language = st.selectbox("Select Output Language", ["English", "French", "Spanish", "German"])

                if user_question:
                    with st.spinner("Searching and generating answer..."):
                        # Generate response
                        response = qa_chain.run(user_question)
                        # Translate the response
                        translated_response = translate_text(response, target_language)

                        # Display results
                        st.write("### Original Answer (English):")
                        st.write(response)
                        st.write(f"### Translated Answer ({target_language}):")
                        st.write(translated_response)
            else:
                st.error("Could not process the document. Please check the file format.")

if __name__ == "__main__":
    main()
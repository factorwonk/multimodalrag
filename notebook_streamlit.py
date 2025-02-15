import streamlit as st
import json
import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List
import nest_asyncio
from subprocess import Popen
import sys
from openai import OpenAI
from operator import itemgetter
from IPython.display import Markdown

def create_streamlit_app():
    """Create the Streamlit app files and supporting modules"""
    
    # Create the main RAG system module
    rag_system_code = """
import json
import os
import uuid
from typing import Dict, Any, List
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

def table_to_string(table):
    \"\"\"Convert table to string representation.\"\"\"
    return "\\n".join(" | ".join(str(cell) for cell in row) for row in table)

def setup_summarization_chain():
    \"\"\"Set up the summarization chain for text and table content.\"\"\"
    prompt_text = \"\"\"
    You are an assistant tasked with summarizing tables and text from financial documents for semantic retrieval.
    These summaries will be embedded and used to retrieve the raw text or table elements.
    Give a detailed summary of the table or text below that is well optimized for retrieval.
    For any tables also add in a description of what the table is about besides the summary.
    Then, include the table in markdown format. Do not add additional words like Summary: etc.

    Table or text chunk:
    {element}
    \"\"\"
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chatgpt = ChatOpenAI(model="gpt-4", temperature=0)
    
    return (
        {"element": RunnablePassthrough()}
        | prompt
        | chatgpt
        | StrOutputParser()
    )

def create_chunks(text: str, chunk_size: int = 4000, chunk_overlap: int = 500) -> List[str]:
    \"\"\"Create chunks from text using RecursiveCharacterTextSplitter.\"\"\"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    return text_splitter.split_text(text)

def process_documents(json_data: List[Dict]) -> tuple:
    \"\"\"Process JSON data and return text and table documents with their summaries.\"\"\"
    processed_data = []
    for entry in json_data:
        # Check if qa field exists and is not empty
        qa_field = entry.get("qa")
        if qa_field is None or qa_field == {}:
            continue
        
        text_parts = []
        if entry.get("pre_text"):
            text_parts.append(" ".join(entry["pre_text"]))
        if entry.get("post_text"):
            text_parts.append(" ".join(entry["post_text"]))
        
        table = entry.get("table_ori", "")
        table_str = table_to_string(table) if table else ""
            
        full_text = "\\n\\n".join(text_parts)
            
        processed_data.append({
            "id": entry.get("id"),
            "text": full_text,
            "table": table_str,
            "qa": entry.get("qa", {})
        })
    
    text_docs = []
    table_docs = []
    
    summarize_chain = setup_summarization_chain()
    
    for item in processed_data:
        if item["text"]:
            chunks = create_chunks(item["text"])
            for chunk in chunks:
                text_docs.append(Document(
                    page_content=chunk,
                    metadata={"parent_id": item["id"]}
                ))
        
        if item["table"]:
            table_docs.append(Document(
                page_content=str(item["table"]),
                metadata={"parent_id": item["id"], "is_table": True}
            ))
    
    text_summaries = summarize_chain.batch([doc.page_content for doc in text_docs], {"max_concurrency": 5})
    table_summaries = summarize_chain.batch([doc.page_content for doc in table_docs], {"max_concurrency": 5})
    
    return text_docs, text_summaries, table_docs, table_summaries

def split_content_types(docs):
    \"\"\"Split retrieved documents into text and table content.\"\"\"
    texts = []
    tables = []
    
    for doc in docs:
        if doc.metadata.get("is_table", False):
            tables.append(doc.page_content)
        else:
            texts.append(doc.page_content)
    
    return {"texts": texts, "tables": tables}

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables):
    \"\"\"Create retriever that indexes summaries but returns raw content.\"\"\"
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    
    return retriever

def prompt_function(data_dict):
    \"\"\"Create a prompt with text and table context.\"\"\"
    formatted_texts = "\\n".join(data_dict["context"]["texts"])
    formatted_tables = "\\n".join(data_dict["context"]["tables"])
    
    prompt_text = f\"\"\"You are an analyst tasked with understanding detailed information from text documents and data tables.
        Use the provided context information to answer the user's question.
        Do not make up answers, use only the provided context documents below.

        User question:
        {data_dict['question']}

        Text context:
        {formatted_texts}

        Table context:
        {formatted_tables}

        Answer:
    \"\"\"
    
    return [HumanMessage(content=prompt_text)]

def setup_rag_chain(retriever, chatgpt):
    \"\"\"Set up the RAG chain.\"\"\"
    rag_chain = (
        {
            "context": itemgetter('context'),
            "question": itemgetter('input'),
        }
        | RunnableLambda(prompt_function)
        | chatgpt
        | StrOutputParser()
    )

    retrieve_docs = (
        itemgetter('input')
        | retriever
        | RunnableLambda(split_content_types)
    )

    return (
        RunnablePassthrough.assign(context=retrieve_docs)
        .assign(answer=rag_chain)
    )

def initialize_rag():
    \"\"\"Initialize the RAG system.\"\"\"
    # Initialize models
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    chatgpt = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Initialize Chroma vectorstore
    vectorstore = Chroma(
        collection_name="text_table_rag",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    
    # Load and process data
    with open('./data/convfinqatrain.json', 'r') as f:
        json_data = json.load(f)
    test_data = json_data[:10]
    
    # Process documents and generate summaries
    text_docs, text_summaries, table_docs, table_summaries = process_documents(test_data)
    
    # Create retriever
    retriever = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        text_docs,
        table_summaries,
        table_docs
    )

    # Setup RAG chain
    rag_chain = setup_rag_chain(retriever, chatgpt)
    
    return rag_chain, retriever

def rag_qa(chain, query: str) -> Dict[str, Any]:
    \"\"\"Execute RAG QA and return response.\"\"\"
    try:
        response = chain.invoke({'input': query})
        return response
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "context": {"texts": [], "tables": []}
        }
"""

    # Create the Streamlit app code
    streamlit_app_code = """
import streamlit as st
import json
import os
from typing import List, Dict, Any
from rag_system import initialize_rag, rag_qa

def format_sources(sources: Dict[str, List[str]]) -> str:
    \"\"\"Format source documents for display in a readable format.\"\"\"
    formatted_text = \"\"
    
    if sources['texts']:
        formatted_text += \"### Text Sources:\\n\"
        for source in sources['texts']:
            formatted_text += f\"- {source}\\n\\n\"
    
    if sources['tables']:
        formatted_text += \"### Table Sources:\\n\"
        for table in sources['tables']:
            formatted_text += f\"```\\n{table}\\n```\\n\\n\"
            
    return formatted_text

def initialize_session_state():
    \"\"\"Initialize the session state variables needed for the application.\"\"\"
    if 'rag_chain' not in st.session_state:
        with st.spinner('Loading RAG model...'):
            try:
                st.session_state.rag_chain, st.session_state.retriever = initialize_rag()
                st.success("RAG model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG model: {str(e)}")
                return False
    
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    if 'question' not in st.session_state:
        st.session_state.question = ''
    
    return True

def render_sidebar():
    \"\"\"Render the application sidebar with helpful information.\"\"\"
    with st.sidebar:
        st.header("About")
        st.markdown(\"\"\"
            This application allows you to query financial data using natural language.
            The system combines text and table information to provide accurate answers.
            
            ### Features:
            - Natural language queries
            - Combined text and table analysis
            - Source document tracking
            
            ### Sample Questions:
            - What was the percentage change in the net cash from operating activities from 2008 to 2009?
            - What was the percent of the growth in the revenues from 2007 to 2008?
            - What was the percent of the growth in the revenues from 2007 to 2008?
            
            ### Tips:
            - Be specific with dates and metrics
            - Use complete sentences
            - Check source documents for context
        \"\"\")


def handle_form_submission():
    \"\"\"Handle the form submission and question processing.\"\"\"
    # Get the current question from session state
    query = st.session_state.question
    
    if not query:
        return
        
    with st.spinner('Processing question...'):
        try:
            # Send query to RAG system and get response
            response = rag_qa(st.session_state.rag_chain, query)
            
            # Add Q&A pair to history
            st.session_state.history.append({
                \"question\": query,
                \"answer\": response['answer'],
                \"sources\": response['context']
            })
            
            # Reset the question in session state by using a new key
            st.session_state.question = \"\"
            
        except Exception as e:
            st.error(f\"An error occurred: {str(e)}\")


def render_input_section():
    \"\"\"Render the query input section with a text field and submit button.\"\"\"
    # Create a form to group input elements
    with st.form(key=\"query_form\"):
        # Create a two-column layout
        col1, col2 = st.columns([4, 1])
        
        # Text input in the first (wider) column
        with col1:
            st.text_input(
                \"Enter your question about the financial data:\",
                key=\"question\",
                placeholder=\"e.g., What was the net change in sales from 2007 to 2008?\"
            )
        
        # Submit button in the second (narrower) column
        with col2:
            submit_button = st.form_submit_button(
                \"Submit\", 
                use_container_width=True,
                on_click=handle_form_submission
            )


def render_history():
    \"\"\"Render the conversation history with questions, answers, and sources.\"\"\"
    if st.session_state.history:
        st.write("---")
        st.write("### Question History")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {item['question']}", expanded=(i == 0)):
                st.write("#### Answer:")
                st.markdown(item['answer'])
                
                st.write("#### Sources:")
                st.markdown(format_sources(item['sources']))


def main():
    st.set_page_config(
        page_title="Financial Data QA System",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("📊 Financial Document QA System")
    
    if not initialize_session_state():
        st.stop()
    
    # Render all comments
    render_sidebar()
    render_input_section()
    render_history()
    
    st.markdown("---")
    st.markdown(
        "💡 Tip: For best results, be specific in your questions and include relevant time periods."
    )

if __name__ == "__main__":
    main()
"""

    # Save the RAG system module
    with open("rag_system.py", "w") as f:
        f.write(rag_system_code)
    
    # Save the Streamlit app
    with open("streamlit_app.py", "w") as f:
        f.write(streamlit_app_code)
    
    print("Created rag_system.py and streamlit_app.py")

def init_notebook():
    """Initialize the notebook environment"""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('chroma_db', exist_ok=True)
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('OPENAI_API_KEY=your-key-here\n')
        print("Created .env file. Please add your OpenAI API key.")
    
    # Create Streamlit app files
    create_streamlit_app()
    
    print("Initialization complete. Ready to start Streamlit app.")

def start_streamlit():
    """Start Streamlit server in a separate process"""
    process = Popen([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "streamlit_app.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])
    return process

def launch_rag_system():
    """
    Launch the RAG system from a Jupyter notebook.
    Returns the Streamlit process which can be terminated later.
    
    Usage:
        process = launch_rag_system()
        # Use the system...
        # When done:
        process.terminate()
    """
    # Initialize the notebook environment
    init_notebook()
    
    # Start Streamlit
    process = start_streamlit()
    
    print("RAG System launched!")
    print("Access the application at: http://localhost:8501")
    print("\nTo stop the server later, use:")
    print("process.terminate()")
    
    return process
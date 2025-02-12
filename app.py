import streamlit as st
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="Financial Data QA System",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def initialize_qa_system():
    """Initialize the QA system with all necessary components"""
    try:
        # Load environment variables
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            return False

        # Load data
        with open('./data/convfinqatrain.json', 'r') as f:
            json_data = json.load(f)
        data = json_data[:10]  # Adjust as needed

        # Initialize embedding model
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Load existing ChromaDB
        persist_directory = "chroma_db"
        if os.path.exists(persist_directory):
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
        else:
            st.error("Vector store not found. Please run the initialization script first.")
            return False

        # Create custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        You are a financial analyst assistant. Use the following pieces of context to answer the question about financial data. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Provide a detailed answer with numerical calculations when applicable:
        """

        PROMPT = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Initialize QA chain
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        return True

    except Exception as e:
        st.error(f"Error initializing QA system: {str(e)}")
        return False

def query_financial_data(question: str) -> Dict:
    """Query the financial data using the QA chain"""
    try:
        result = st.session_state.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "source_documents": []
        }

# Main app layout
st.title("📊 Financial Data QA System")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
        This application allows you to query financial data using natural language.
        Simply type your question about financial metrics, trends, or specific data points.
        
        Example questions:
        - What was the net change in sales from 2008 to 2009?
        - How did the profit margins evolve over time?
        - What were the total assets in the most recent year?
    """)
    
    # Initialize button
    if st.button("Initialize/Reset System"):
        with st.spinner("Initializing QA system..."):
            if initialize_qa_system():
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system.")

# Main content
if st.session_state.qa_chain is None:
    st.info("Please initialize the system using the button in the sidebar.")
else:
    # Query input
    question = st.text_input(
        "Enter your financial question:",
        placeholder="e.g., What was the net change in sales from 2008 to 2009?"
    )

    # Process query
    if question:
        with st.spinner("Processing your question..."):
            result = query_financial_data(question)
            
            # Display answer
            st.markdown("### Answer")
            st.write(result["answer"])
            
            # Display source documents
            with st.expander("View Source Documents"):
                for idx, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Source Document {idx}**")
                    st.markdown(f'<div class="source-box">{doc}</div>', unsafe_allow_html=True)

    # Additional features
    with st.expander("Advanced Options"):
        st.markdown("### Query History")
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if question and question not in st.session_state.query_history:
            st.session_state.query_history.append(question)
        
        for historic_query in st.session_state.query_history:
            st.write(f"- {historic_query}")

# Footer
st.markdown("---")
st.markdown(
    "💡 Tip: For best results, be specific in your questions and include relevant time periods or metrics."
)
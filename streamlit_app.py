
import streamlit as st
import json
import os
from typing import List, Dict, Any
from rag_system import initialize_rag, rag_qa

def format_sources(sources: Dict[str, List[str]]) -> str:
    """Format source documents for display in a readable format."""
    formatted_text = ""
    
    if sources['texts']:
        formatted_text += "### Text Sources:\n"
        for source in sources['texts']:
            formatted_text += f"- {source}\n\n"
    
    if sources['tables']:
        formatted_text += "### Table Sources:\n"
        for table in sources['tables']:
            formatted_text += f"```\n{table}\n```\n\n"
            
    return formatted_text

def initialize_session_state():
    """Initialize the session state variables needed for the application."""
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
    """Render the application sidebar with helpful information."""
    with st.sidebar:
        st.header("About")
        st.markdown("""
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
        """)


def handle_form_submission():
    """Handle the form submission and question processing."""
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
                "question": query,
                "answer": response['answer'],
                "sources": response['context']
            })
            
            # Reset the question in session state by using a new key
            st.session_state.question = ""
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def render_input_section():
    """Render the query input section with a text field and submit button."""
    # Create a form to group input elements
    with st.form(key="query_form"):
        # Create a two-column layout
        col1, col2 = st.columns([4, 1])
        
        # Text input in the first (wider) column
        with col1:
            st.text_input(
                "Enter your question about the financial data:",
                key="question",
                placeholder="e.g., What was the net change in sales from 2007 to 2008?"
            )
        
        # Submit button in the second (narrower) column
        with col2:
            submit_button = st.form_submit_button(
                "Submit", 
                use_container_width=True,
                on_click=handle_form_submission
            )


def render_history():
    """Render the conversation history with questions, answers, and sources."""
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

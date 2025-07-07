

from helper_functions import *


def render_document_cards(results: List[Dict]):

    cols = st.columns(3)
    
    for idx, result in enumerate(results[:3]):
        with cols[idx]:
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; text-align:center;">
                <h4>{['First', 'Second', 'Third'][idx]} Result</h4>
                <p><strong>{result['source_file']}</strong></p>
                <p>Page: {result['page']}</p>
                <small>Score: {result['similarity_score']:.4f}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"View Details", key=f"view_{idx}"):
                st.session_state.selected_doc = idx

def render_detailed_view(query: str, results: List[Dict], model: SentenceTransformer):

    if 'selected_doc' not in st.session_state:
        return None
    
    idx = st.session_state.selected_doc
    if idx is None or idx >= len(results):
        return None
    
    result = results[idx]
    text = result['text']
    
    st.markdown(f"### Document Details - Rank {idx + 1}")
    
    # Highlight similar words
    similar_words = get_similar_words(query, text, model)
    highlighted_text = highlight_text(text, similar_words[:5])
    
    # Display content in a container
    with st.container():
        st.markdown("**Content:**")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.write(f"**Similarity Score:** {result['similarity_score']:.4f}")
        st.write(f"**Top Similar Words:** {', '.join(similar_words[:5])}")
    
    return text

def render_ai_section(query: str, context: str, dialog_model, tokenizer):

    st.markdown("---")
    st.markdown("### 🧠 AI Answer Generation")
    
    ai_query = st.text_area("Edit your question:", value=query, height=100)
    
    if st.button("Generate AI Answer", use_container_width=True):
        if ai_query.strip():
            with st.spinner("Generating answer..."):
                answer = generate_ai_answer(ai_query, context, dialog_model, tokenizer)
            
            st.markdown("### AI Answer")
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; border-left:5px solid #28a745;">
                <p style="color:#333; line-height:1.6; margin:0;">{answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            
def render_database_summary_page(summary_data: Dict):

    st.title("📊 Database Summary")
    
    # Extract summary list
    summaries = summary_data.get('summary', [])
    
    # Overview metrics
    st.markdown("## 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", len(summaries))
    with col2:
        # Count unique models
        unique_models = len(set(item['model'] for item in summaries))
        st.metric("🤖 Unique Models", unique_models)
    with col3:
        # Count unique QA models
        unique_qa_models = len(set(item['qa_model'] for item in summaries))
        st.metric("🧠 QA Models", unique_qa_models)
    with col4:
        # Count processed summaries (not TODO)
        processed_count = sum(1 for item in summaries if item['summary'] != 'TODO')
        st.metric("✅ Processed", processed_count)
    
    st.markdown("---")
    
    # Model information
    st.markdown("## 🤖 Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Summarization Models")
        model_counts = {}
        for item in summaries:
            model = item['model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        for model, count in model_counts.items():
            st.write(f"**{model}:** {count} documents")
    
    with col2:
        st.markdown("### QA Models")
        qa_model_counts = {}
        for item in summaries:
            qa_model = item['qa_model']
            qa_model_counts[qa_model] = qa_model_counts.get(qa_model, 0) + 1
        
        for qa_model, count in qa_model_counts.items():
            st.write(f"**{qa_model}:** {count} documents")
    

    
    st.markdown("---")
    
    # Document Details
    st.markdown("## 📁 Document Details")
    
    if summaries:
        # Create expandable sections for each document
        for i, item in enumerate(summaries, 1):
            # Status indicator
            status_icon = "✅" if item['summary'] != 'TODO' else "⏳"
            
            with st.expander(f"{status_icon} {i}. {item['source_file']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Source File:** `{item['source_file']}`")
                    st.write(f"**Summary Model:** {item['model']}")
                    #st.write(f"**QA Model:** {item['qa_model']}")
                
                with col2:
                    st.write(f"**Device:** {item['device'].strip()}")
                    if item['summary'] != 'TODO':
                        st.write(f"**Status:** ✅ Processed")
                    else:
                        st.write(f"**Status:** ⏳ Pending")
                
                # Summary section
                st.markdown("**Summary:**")
                if item['summary'] != 'TODO':
                    st.write(item['summary'])
                else:
                    st.info("Summary not yet generated")
    
    st.markdown("---")
    

    with st.expander("🔍 Raw Database Info (Advanced)"):
        st.markdown("### Summary Data Structure")
        st.json(summary_data)
        
        st.markdown("### Sample Document Entry")
        if summaries:
            st.json(summaries[0])  # Show first entry as example

def render_database_summary(summary_data: Dict):

    st.title("📊 Database Summary")
    
    summaries = summary_data.get('summary', [])
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", len(summaries))
    with col2:
        processed = sum(1 for s in summaries if s.get('summary', 'TODO') != 'TODO')
        st.metric("✅ Processed", processed)

    with col3:
        if summaries:
            st.metric("Progress", f"{processed/len(summaries):.1%}")
    
    # Document list
    st.markdown("## 📁 Documents")
    for i, item in enumerate(summaries, 1):
        status = "✅" if item.get('summary', 'TODO') != 'TODO' else "⏳"
        
        with st.expander(f"{status} {i}. {item['source_file']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File:** {item['source_file']}")
                st.write(f"**Model:** {item.get('model', 'N/A')}")
            with col2:
                st.write(f"**Device:** {item.get('device', 'N/A')}")
                st.write(f"**Status:** {'Processed' if status == '✅' else 'Pending'}")
            
            if item.get('summary', 'TODO') != 'TODO':
                st.markdown("**Summary:**")
                st.write(item['summary'])

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def setup_sidebar():

    with st.sidebar:
        st.markdown("# 🧭 Navigation")
        st.markdown("---")
        
        # Navigation buttons
        if st.button("🔍 Document Search", use_container_width=True):
            st.session_state.current_page = "search"
        
        if st.button("📊 Database Summary", use_container_width=True):
            st.session_state.current_page = "summary"
        
        # Initialize current_page if not exists
        if "current_page" not in st.session_state:
            st.session_state.current_page = "search"
        
        return st.session_state.current_page

def main():

    st.set_page_config(
        page_title="Document Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "search"
    
    # Setup navigation
    current_page = setup_sidebar()
    
    # Load models and data
    sentence_model, dialog_model, tokenizer = load_models()
    index, metadata, summary_data = load_database()
    
    # Route to pages
    if current_page == "search":
        st.title("🔍 Document Search")
        
        # Query input
        query = st.text_input("Enter your search query:")
        if query == '1':  # Shortcut for default query
            query = DEFAULT_QUERY
        
        if query:
            st.write(f"**Searching for:** '{query}'")
            
            # Search and display results
            results = search_documents(query, sentence_model, index, metadata)
            
            st.markdown("### Top 3 Matches")
            render_document_cards(results)
            
            # Detailed view
            context = render_detailed_view(query, results, sentence_model)
            
            # AI answer section
            if context:
                render_ai_section(query, context, dialog_model, tokenizer)
    
    elif current_page == "summary":
        render_database_summary_page(summary_data)

if __name__ == "__main__":
    main()
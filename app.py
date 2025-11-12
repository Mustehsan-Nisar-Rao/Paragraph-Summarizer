# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import time

# Set page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the summarization model from Hugging Face"""
    try:
        model_name = "mustehsannisarrao/summarizer"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try to load the model directly
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        st.sidebar.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_summary(tokenizer, model, text, max_length=128, min_length=30):
    """Generate summary for input text"""
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            max_length=512, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Using fine-tuned BART model from Hugging Face")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a fine-tuned BART model for text summarization. "
        "The model is loaded directly from Hugging Face Hub."
    )
    
    st.sidebar.title("Model Information")
    st.sidebar.markdown("""
    - **Model**: mustehsannisarrao/summarizer
    - **Base Model**: BART
    - **Task**: Text Summarization
    - **Framework**: PyTorch + Hugging Face
    """)
    
    st.sidebar.title("Parameters")
    max_length = st.sidebar.slider("Max Summary Length", 50, 200, 128)
    min_length = st.sidebar.slider("Min Summary Length", 10, 100, 30)
    
    # Load model
    with st.spinner("Loading summarization model from Hugging Face..."):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load the model. Please check your internet connection.")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Processing", "Examples"])
    
    with tab1:
        st.header("Summarize Single Text")
        
        # Text input
        text_input = st.text_area(
            "Enter your text to summarize:",
            placeholder="Paste your article, document, or any long text here...",
            height=200
        )
        
        # Example texts
        example_texts = {
            "News Article": "The company reported strong earnings this quarter with profits increasing by 20%. This growth was driven by successful product launches and expanding market share in emerging economies. The CEO expressed optimism about future growth prospects and announced plans for further international expansion.",
            "Research Abstract": "Researchers have discovered a new species of marine life in the deep ocean. The creature exhibits unique bioluminescent properties that enable communication in complete darkness. This discovery could lead to significant advances in biomedical imaging technology and underwater communication systems.",
            "Product Review": "The new smartphone features a revolutionary camera system that outperforms all competitors in the market. With advanced AI processing, enhanced low-light capabilities, and improved image stabilization, it sets a new standard for mobile photography. The device also offers extended battery life and will be available starting next month."
        }
        
        # Example selector
        selected_example = st.selectbox("Or choose an example:", ["None"] + list(example_texts.keys()))
        if selected_example != "None":
            text_input = example_texts[selected_example]
        
        # Generate button
        if st.button("Generate Summary", type="primary"):
            if text_input.strip():
                with st.spinner("Generating summary..."):
                    start_time = time.time()
                    summary = generate_summary(tokenizer, model, text_input, max_length, min_length)
                    end_time = time.time()
                    
                    if summary:
                        # Display results
                        st.subheader("📄 Generated Summary")
                        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown('<div class="metric-box">'
                                       '<h3>📊</h3>'
                                       f'<b>Original Length</b><br>{len(text_input)} chars'
                                       '</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-box">'
                                       '<h3>📝</h3>'
                                       f'<b>Summary Length</b><br>{len(summary)} chars'
                                       '</div>', unsafe_allow_html=True)
                        
                        with col3:
                            compression = (1 - len(summary)/len(text_input)) * 100
                            st.markdown('<div class="metric-box">'
                                       '<h3>⚡</h3>'
                                       f'<b>Compression</b><br>{compression:.1f}%'
                                       '</div>', unsafe_allow_html=True)
                        
                        with col4:
                            processing_time = end_time - start_time
                            st.markdown('<div class="metric-box">'
                                       '<h3>⏱️</h3>'
                                       f'<b>Processing Time</b><br>{processing_time:.2f}s'
                                       '</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to summarize.")
    
    with tab2:
        st.header("Batch Processing")
        
        st.subheader("Enter multiple texts (one per line):")
        batch_text = st.text_area(
            "Batch texts:",
            placeholder="Enter each text on a new line...\n\nExample:\nFirst article text...\nSecond article text...\nThird article text...",
            height=250,
            key="batch_input"
        )
        
        if st.button("Process Batch", key="batch_process"):
            if batch_text.strip():
                texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                
                if texts:
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, text in enumerate(texts):
                        summary = generate_summary(tokenizer, model, text, max_length, min_length)
                        if summary:
                            results.append({
                                'Original': text[:100] + "..." if len(text) > 100 else text,
                                'Summary': summary
                            })
                        progress_bar.progress((i + 1) / len(texts))
                    
                    # Display results
                    if results:
                        st.subheader("Batch Results")
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1}: {result['Original']}"):
                                st.write("**Original:**", result['Original'])
                                st.write("**Summary:**", result['Summary'])
                    else:
                        st.error("No summaries were generated successfully.")
                else:
                    st.warning("Please enter at least one valid text.")
            else:
                st.warning("Please enter some texts to process.")
    
    with tab3:
        st.header("Example Usage")
        
        st.markdown("""
        ### How to use this model in your code:
        
        ```python
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Load the model
        model_name = "mustehsannisarrao/summarizer"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Generate summary
        text = "Your long article text here..."
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
        
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=128,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(summary)
        ```
        
        ### Model Details:
        - **Hugging Face**: [mustehsannisarrao/summarizer](https://huggingface.co/mustehsannisarrao/summarizer)
        - **Base Architecture**: BART
        - **Fine-tuned for**: Text Summarization
        - **Input**: Long text documents
        - **Output**: Concise summaries
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Model Repository**: [mustehsannisarrao/summarizer](https://huggingface.co/mustehsannisarrao/summarizer) | "
        "**Built with**: Streamlit & Hugging Face Transformers"
    )

if __name__ == "__main__":
    main()

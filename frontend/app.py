"""
Multi-modal AI Agent Frontend

A beautiful Streamlit interface for the multi-modal content summarization service.
Provides an intuitive way to input URLs, view results, and interact with the AI agent.
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Multi-modal AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SUMMARY_TYPES = [
    "comprehensive",
    "executive", 
    "bullet_points",
    "key_takeaways",
    "visual_focused"
]


class MultiModalFrontend:
    """Frontend application for the Multi-modal AI Agent."""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def check_api_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def process_url(self, url: str, summary_type: str, max_length: int, 
                   include_visual_insights: bool, detect_contradictions: bool) -> Dict[str, Any]:
        """Process a URL through the complete pipeline."""
        payload = {
            "url": url,
            "summary_type": summary_type,
            "max_length": max_length,
            "include_visual_insights": include_visual_insights,
            "detect_contradictions": detect_contradictions
        }
        
        response = self.session.post(
            f"{self.api_base_url}/process-url",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def scrape_content(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL."""
        payload = {"url": url, "headless": True}
        
        response = self.session.post(
            f"{self.api_base_url}/scrape",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Scraping Error: {response.status_code} - {response.text}")
    
    def analyze_images(self, image_urls: List[str], text_context: str) -> List[Dict[str, Any]]:
        """Analyze images."""
        payload = {
            "image_urls": image_urls,
            "text_context": text_context
        }
        
        response = self.session.post(
            f"{self.api_base_url}/analyze",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Analysis Error: {response.status_code} - {response.text}")


def main():
    """Main application function."""
    
    # Initialize frontend
    frontend = MultiModalFrontend()
    
    # Header
    st.markdown('<h1 class="main-header"> Multi-modal AI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced content summarization with text and visual analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Health Check
        api_healthy = frontend.check_api_health()
        if api_healthy:
            st.success("API Connected")
        else:
            st.error("API Unavailable")
            st.info("Please ensure the backend API is running on the configured URL.")
        
        st.divider()
        
        # Summary Type Selection
        st.subheader("Summary Type")
        summary_type = st.selectbox(
            "Choose summary format:",
            SUMMARY_TYPES,
            index=0,
            help="Select the type of summary you want to generate"
        )
        
        # Summary Length
        st.subheader("📏 Summary Length")
        max_length = st.slider(
            "Maximum words:",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum number of words in the summary"
        )
        
        # Options
        st.subheader("🔧 Options")
        include_visual_insights = st.checkbox(
            "Include visual insights",
            value=True,
            help="Include analysis of images and visual content"
        )
        
        detect_contradictions = st.checkbox(
            "Detect contradictions",
            value=True,
            help="Check for contradictions between text and visual content"
        )
        
        st.divider()
        
        # Cache Management
        st.subheader("Cache")
        if st.button("Clear Cache"):
            try:
                response = frontend.session.get(f"{frontend.api_base_url}/cache/clear")
                if response.status_code == 200:
                    st.success("Cache cleared successfully!")
                else:
                    st.error("Failed to clear cache")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
        
        # Cache Stats
        try:
            response = frontend.session.get(f"{frontend.api_base_url}/cache/stats")
            if response.status_code == 200:
                stats = response.json()
                st.metric("Cached Items", stats.get("total_keys", 0))
        except Exception:
            pass
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Quick Process", "Step-by-Step", "Results History"])
    
    with tab1:
        st.header("Quick URL Processing")
        st.write("Enter a URL to get a comprehensive multi-modal summary in one step.")
        
        # URL input
        url = st.text_input(
            "Enter URL:",
            placeholder="https://example.com/article",
            help="Enter the URL of the article or webpage you want to summarize"
        )
        
        if st.button("Process URL", type="primary", use_container_width=True):
            if url:
                if not url.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    process_url_quick(frontend, url, summary_type, max_length, 
                                   include_visual_insights, detect_contradictions)
            else:
                st.error("Please enter a URL")
    
    with tab2:
        st.header("Step-by-Step Processing")
        st.write("Process content step by step for more control over each stage.")
        
        # Step 1: Scraping
        st.subheader("Step 1: Content Scraping")
        scrape_url = st.text_input(
            "URL to scrape:",
            placeholder="https://example.com/article",
            key="scrape_url"
        )
        
        if st.button("Scrape Content", key="scrape_btn"):
            if scrape_url:
                if not scrape_url.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL")
                else:
                    scrape_content_step(frontend, scrape_url)
            else:
                st.error("Please enter a URL")
        
        # Step 2: Image Analysis
        st.subheader("Step 2: Image Analysis")
        if 'scraped_content' in st.session_state:
            image_urls = [img['url'] for img in st.session_state.scraped_content.get('images', [])]
            if image_urls:
                st.write(f"Found {len(image_urls)} images to analyze")
                if st.button("Analyze Images"):
                    analyze_images_step(frontend, image_urls, st.session_state.scraped_content.get('text_content', ''))
            else:
                st.info("No images found in the scraped content")
        
        # Step 3: Summarization
        st.subheader("Step 3: Generate Summary")
        if 'image_analysis' in st.session_state and 'scraped_content' in st.session_state:
            if st.button("Generate Summary"):
                generate_summary_step(frontend, st.session_state.scraped_content.get('text_content', ''),
                                   st.session_state.image_analysis, summary_type, max_length,
                                   include_visual_insights, detect_contradictions)
    
    with tab3:
        st.header("Results History")
        st.write("View and manage your previous processing results.")
        
        # This would typically connect to a database
        st.info("Results history feature coming soon!")
        
        # Placeholder for results display
        if 'processing_results' in st.session_state:
            display_results_history(st.session_state.processing_results)


def process_url_quick(frontend: MultiModalFrontend, url: str, summary_type: str, 
                     max_length: int, include_visual_insights: bool, detect_contradictions: bool):
    """Process a URL quickly with progress tracking."""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        progress_bar.progress(10)
        status_text.text("🔄 Initializing...")
        
        # Process URL
        status_text.text("🔄 Processing URL (this may take a few minutes)...")
        result = frontend.process_url(url, summary_type, max_length, 
                                    include_visual_insights, detect_contradictions)
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Store result in session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = []
        st.session_state.processing_results.append({
            'timestamp': datetime.now(),
            'url': url,
            'result': result
        })
        
        # Display results
        display_processing_results(result)
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("Processing failed!")
        st.error(f"Error processing URL: {str(e)}")
    
    finally:
        # Clear progress indicators after a delay
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()


def scrape_content_step(frontend: MultiModalFrontend, url: str):
    """Step 1: Scrape content from URL."""
    
    with st.spinner("Scraping content..."):
        try:
            result = frontend.scrape_content(url)
            st.session_state.scraped_content = result
            
            # Display scraped content summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", f"{len(result.get('text_content', '')):,} chars")
            with col2:
                st.metric("Images Found", len(result.get('images', [])))
            with col3:
                st.metric("Title", result.get('title', 'N/A')[:30] + "..." if len(result.get('title', '')) > 30 else result.get('title', 'N/A'))
            
            st.success("Content scraped successfully!")
            
            # Show preview
            with st.expander("Content Preview"):
                st.text_area("Text Content", result.get('text_content', '')[:1000] + "...", height=200)
                
                if result.get('images'):
                    st.write("**Images Found:**")
                    for i, img in enumerate(result['images'][:5]):  # Show first 5 images
                        st.write(f"{i+1}. {img.get('url', 'N/A')}")
            
        except Exception as e:
            st.error(f"Error scraping content: {str(e)}")


def analyze_images_step(frontend: MultiModalFrontend, image_urls: List[str], text_context: str):
    """Step 2: Analyze images."""
    
    with st.spinner("Analyzing images..."):
        try:
            results = frontend.analyze_images(image_urls, text_context)
            st.session_state.image_analysis = results
            
            # Display analysis summary
            successful = len([r for r in results if 'error' not in r])
            st.metric("Successful Analyses", f"{successful}/{len(results)}")
            
            st.success("Image analysis complete!")
            
            # Show analysis results
            with st.expander("Analysis Results"):
                for i, result in enumerate(results):
                    if 'error' not in result:
                        st.write(f"**Image {i+1}:**")
                        st.write(f"- Caption: {result.get('caption', {}).get('caption', 'N/A')}")
                        st.write(f"- OCR Text: {result.get('ocr_text', {}).get('text', 'N/A')[:100]}...")
                        st.write(f"- Chart Type: {result.get('chart_type', {}).get('chart_type', 'N/A')}")
                        st.write(f"- Relevance: {result.get('relevance_score', 0):.2f}")
                        st.divider()
            
        except Exception as e:
            st.error(f"Error analyzing images: {str(e)}")


def generate_summary_step(frontend: MultiModalFrontend, text_content: str, image_analysis: List[Dict[str, Any]],
                         summary_type: str, max_length: int, include_visual_insights: bool, detect_contradictions: bool):
    """Step 3: Generate summary."""
    
    with st.spinner("Generating summary..."):
        try:
            # This would call the summarize endpoint
            # For now, we'll simulate the result
            st.success("Summary generated successfully!")
            st.info("Summary generation step completed. Full integration with summarize endpoint coming soon!")
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")


def display_processing_results(result: Dict[str, Any]):
    """Display comprehensive processing results."""
    
    st.header("Processing Results")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processing Time", f"{result.get('processing_time', 0):.1f}s")
    with col2:
        st.metric("Text Length", f"{result.get('scraped_content', {}).get('text_length', 0):,} chars")
    with col3:
        st.metric("Images Analyzed", result.get('image_analysis', {}).get('successful_analyses', 0))
    with col4:
        confidence = result.get('summary', {}).get('confidence_score', 0)
        st.metric("Confidence", f"{confidence:.2%}")
    
    # Summary
    st.subheader("Summary")
    summary_text = result.get('summary', {}).get('text', '')
    if summary_text:
        st.markdown(summary_text)
    else:
        st.warning("No summary generated")
    
    # Key Points
    key_points = result.get('summary', {}).get('key_points', [])
    if key_points:
        st.subheader("Key Points")
        for i, point in enumerate(key_points, 1):
            st.write(f"{i}. {point}")
    
    # Visual Insights
    visual_insights = result.get('summary', {}).get('visual_insights', [])
    if visual_insights:
        st.subheader("Visual Insights")
        for insight in visual_insights:
            with st.expander(f"Image: {insight.get('image_url', 'N/A')}"):
                st.write(f"**Type:** {insight.get('insight_type', 'N/A')}")
                st.write(f"**Description:** {insight.get('description', 'N/A')}")
                st.write(f"**Relevance:** {insight.get('relevance_score', 0):.2f}")
    
    # Contradictions
    contradictions = result.get('summary', {}).get('contradictions', [])
    if contradictions:
        st.subheader("Contradictions Detected")
        for contradiction in contradictions:
            st.warning(f"**Image:** {contradiction.get('image_url', 'N/A')}")
            st.write(f"**Description:** {contradiction.get('contradiction_description', 'N/A')}")
    
    # Detailed Analysis
    with st.expander("Detailed Analysis"):
        # Image Analysis Details
        image_results = result.get('image_analysis', {}).get('results', [])
        if image_results:
            st.subheader("Image Analysis Details")
            
            # Create a DataFrame for better visualization
            analysis_data = []
            for img in image_results:
                if 'error' not in img:
                    analysis_data.append({
                        'URL': img.get('url', 'N/A'),
                        'Caption': img.get('caption', {}).get('caption', 'N/A'),
                        'Chart Type': img.get('chart_type', {}).get('chart_type', 'N/A'),
                        'Relevance Score': img.get('relevance_score', 0),
                        'OCR Confidence': img.get('ocr_text', {}).get('confidence', 0)
                    })
            
            if analysis_data:
                df = pd.DataFrame(analysis_data)
                st.dataframe(df)
                
                # Relevance score chart
                if len(df) > 1:
                    fig = px.bar(df, x='URL', y='Relevance Score', 
                               title="Image Relevance Scores")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Model Information
    model_info = result.get('summary', {}).get('model_used', 'N/A')
    token_usage = result.get('summary', {}).get('token_usage', {})
    
    if model_info or token_usage:
        st.subheader("🤖 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Model Used:** {model_info}")
        with col2:
            if token_usage:
                st.write(f"**Tokens Used:** {token_usage.get('total_tokens', 'N/A')}")


def display_results_history(results: List[Dict[str, Any]]):
    """Display results history."""
    
    if not results:
        st.info("No processing history available")
        return
    
    # Create a DataFrame for the history
    history_data = []
    for result in results:
        history_data.append({
            'Timestamp': result['timestamp'],
            'URL': result['url'],
            'Processing Time': result['result'].get('processing_time', 0),
            'Confidence': result['result'].get('summary', {}).get('confidence_score', 0),
            'Images Analyzed': result['result'].get('image_analysis', {}).get('successful_analyses', 0)
        })
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)
    
    # Show processing time trend
    if len(df) > 1:
        fig = px.line(df, x='Timestamp', y='Processing Time', 
                     title="Processing Time Trend")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

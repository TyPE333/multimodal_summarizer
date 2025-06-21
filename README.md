
# Multi-modal AI Agent for Interactive Content Summarization

A comprehensive AI system that leverages both text and image modalities to create engaging and comprehensive content summarization experiences, mimicking the capabilities of advanced multimodal LLMs.

## Pain Point and solution

Summarizing complex online content (articles, reports, research papers) isn't just about text. Visuals (charts, diagrams, key images) often convey critical information. This AI agent can summarize web articles/reports, extract key information from images, and present a consolidated, multi-modal summary.

## Features

### Content Ingestion & Extraction
- **Web Scraping**: Robust scrapers using Playwright for dynamic content
- **Text Preprocessing**: Clean extracted text with boilerplate removal
- **Image Extraction**: Download and process images from web content

### Image Understanding (Computer Vision)
- **OCR Integration**: Extract text from images using EasyOCR
- **Image Captioning**: Generate descriptive captions using CLIP
- **Object Detection**: Identify key objects and elements in images
- **Chart/Diagram Analysis**: Specialized analysis for technical diagrams

### LLM for Multi-modal Summarization
- **Input Aggregation**: Combine text, OCR, and image captions
- **Advanced Prompt Engineering**: Sophisticated prompts for multimodal synthesis
- **Contradiction Detection**: Identify conflicts between text and visual content
- **Structured Output**: Generate bullet points, key takeaways, and explanations

### User Interface & Interaction
- **Web Application**: Interactive Streamlit interface
- **Multi-modal Display**: Present text summaries with image analysis
- **Highlighted Integration**: Show how images inform text summaries

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   ML Services   │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (Docker)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Web Scraper   │
                       │   (Playwright)  │
                       └─────────────────┘
```

## Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **Playwright**: Web scraping and automation
- **EasyOCR**: Optical Character Recognition
- **Transformers**: CLIP for image captioning
- **OpenAI/Anthropic**: LLM integration
- **Pillow**: Image processing
- **BeautifulSoup**: HTML parsing

### Frontend
- **Streamlit**: Interactive web interface
- **Plotly**: Data visualization
- **Pillow**: Image display

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **Redis**: Caching and session management

## Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Git

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd multimodal-summarizer

# Start the application
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
```

### Manual Setup
```bash
# Backend setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend setup
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
```

## Usage

1. **Start the Application**: Run `docker-compose up --build`
2. **Access the Interface**: Navigate to `http://localhost:8501`
3. **Input a URL**: Enter the URL of an article or report
4. **View Results**: Get a comprehensive multimodal summary

## Project Structure

```
multimodal-summarizer/
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── web_scraper.py         # Web scraping functionality
│   ├── image_analyzer.py      # Computer vision processing
│   ├── summarizer.py          # LLM integration and summarization
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile            # Backend container
├── frontend/
│   ├── app.py                # Streamlit application
│   ├── requirements.txt      # Frontend dependencies
│   └── Dockerfile           # Frontend container
├── docker-compose.yml        # Multi-service orchestration
└── README.md                # Project documentation
```

## API Endpoints

### Backend API (FastAPI)

- `POST /scrape`: Scrape content from a URL
- `POST /analyze`: Analyze images and extract information
- `POST /summarize`: Generate multimodal summaries
- `GET /health`: Health check endpoint

### Request/Response Examples

```python
# Scrape content
POST /scrape
{
    "url": "https://example.com/article"
}

# Analyze images
POST /analyze
{
    "image_urls": ["url1", "url2"],
    "text_content": "article text"
}

# Generate summary
POST /summarize
{
    "text_content": "article text",
    "image_analysis": {...},
    "summary_type": "comprehensive"
}
```

## Performance Optimizations

### Latency Reduction
- **Parallel Processing**: Images processed concurrently
- **Caching**: Redis-based caching for repeated requests
- **Model Optimization**: Efficient CV model loading
- **Async Processing**: Non-blocking API operations

### Scalability
- **Container Orchestration**: Docker Compose for easy scaling
- **Load Balancing**: Multiple backend instances
- **Database Caching**: Redis for session and result caching
- **Resource Management**: Efficient memory and CPU usage

## Security Considerations

- **Input Validation**: Sanitize URLs and user inputs
- **Rate Limiting**: Prevent abuse and manage costs
- **API Key Management**: Secure storage of API credentials
- **Error Handling**: Graceful failure handling

## Monitoring & Logging

- **Application Logs**: Structured logging with different levels
- **Performance Metrics**: Response times and resource usage
- **Error Tracking**: Comprehensive error reporting
- **Usage Analytics**: Track API usage and costs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Hugging Face for transformers library
- Microsoft for Playwright
- JaidedAI for EasyOCR

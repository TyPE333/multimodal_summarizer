# Quick Start Guide

Get your Multi-modal AI Agent up and running in minutes!

## Prerequisites

- **Docker & Docker Compose** - [Install here](https://docs.docker.com/get-docker/)
- **Python 3.9+** (for setup script)
- **API Keys** - OpenAI and/or Anthropic (optional but recommended)

## Quick Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multimodal-summarizer
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   The script will:
   - Check prerequisites
   - Create configuration files
   - Prompt for API keys
   - Build and start all services

3. **Access the application**
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## 🔧 Manual Setup

If you prefer manual setup:

1. **Create environment file**
   ```bash
   cp config.env.example .env
   # Edit .env with your API keys
   ```

2. **Build and start services**
   ```bash
   docker-compose up --build -d
   ```

3. **Check service status**
   ```bash
   docker-compose ps
   ```

## First Use

1. **Open the frontend** at http://localhost:8501

2. **Enter a URL** to process:
   - Try a news article with images
   - Or a research paper with charts
   - Or any webpage with visual content

3. **Choose summary type**:
   - **Comprehensive**: Full analysis with visual insights
   - **Executive**: Business-focused summary
   - **Bullet Points**: Key points only
   - **Key Takeaways**: Main insights
   - **Visual Focused**: Emphasis on visual content

4. **Click "Process URL"** and wait for results

## Understanding Results

The system provides:

- **Text Summary**: AI-generated summary of the content
- **Key Points**: Extracted main insights
- **Visual Insights**: Analysis of images, charts, diagrams
- **Contradictions**: Any conflicts between text and visuals
- **Confidence Score**: Reliability of the analysis

## 🔍 API Usage

Use the REST API directly:

```bash
# Process a URL
curl -X POST "http://localhost:8000/process-url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "summary_type": "comprehensive"
  }'

# Scrape content only
curl -X POST "http://localhost:8000/scrape" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Analyze images
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/image.jpg"],
    "text_context": "Article text..."
  }'
```

## 🛠️ Troubleshooting

### Common Issues

1. **Services won't start**
   ```bash
   # Check logs
   docker-compose logs
   
   # Restart services
   docker-compose down && docker-compose up -d
   ```

2. **API keys not working**
   - Verify keys in `.env` file
   - Check API key validity
   - Ensure sufficient credits

3. **Slow processing**
   - First run downloads models (can take 5-10 minutes)
   - Check available memory (4GB+ recommended)
   - Consider using GPU for faster processing

4. **Web scraping fails**
   - Some sites block automated access
   - Try different URLs
   - Check if site requires JavaScript

### Performance Tips

- **Use GPU**: Set `DEVICE=cuda` in `.env` for faster processing
- **Increase memory**: Allocate more RAM to Docker
- **Cache results**: Results are cached to avoid reprocessing
- **Batch processing**: Process multiple URLs efficiently

## Scaling

For production use:

1. **Use production profile**
   ```bash
   docker-compose --profile production up -d
   ```

2. **Configure reverse proxy** (nginx included)

3. **Set up monitoring** and logging

4. **Use external Redis** for persistence

5. **Configure SSL certificates**

## Security

- **API Keys**: Never commit `.env` files
- **Network**: Use internal networks in production
- **Access Control**: Implement authentication
- **Rate Limiting**: Configure request limits

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [API documentation](http://localhost:8000/docs)
- Check out the [examples](examples/) directory
- Join the community for support and updates

## Support

- **Issues**: Create a GitHub issue
- **Documentation**: Check README.md and API docs
- **Community**: Join our discussions
- **Email**: Contact the maintainers

---

**Happy summarizing! 🎉** 
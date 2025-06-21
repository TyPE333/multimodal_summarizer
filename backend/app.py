"""
Multi-modal AI Agent FastAPI Application

This is the main FastAPI application that orchestrates web scraping, image analysis,
and summarization to provide a comprehensive multi-modal content summarization service.
"""

import asyncio
import os
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
import redis.asyncio as redis
from loguru import logger

from web_scraper import WebScraper, scrape_article
from image_analyzer import ImageAnalyzer, analyze_images_batch
from summarizer import MultiModalSummarizer, SummaryRequest, SummaryType


# Pydantic models for API requests/responses
class ScrapeRequest(BaseModel):
    url: HttpUrl
    headless: bool = True
    
    @validator('url')
    def validate_url(cls, v):
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class AnalyzeRequest(BaseModel):
    image_urls: List[HttpUrl]
    text_context: str = ""
    
    @validator('image_urls')
    def validate_image_urls(cls, v):
        if not v:
            raise ValueError('At least one image URL must be provided')
        if len(v) > 20:
            raise ValueError('Maximum 20 images allowed per request')
        return v


class SummarizeRequest(BaseModel):
    text_content: str
    image_analysis: List[Dict[str, Any]]
    summary_type: str = "comprehensive"
    max_length: int = 1000
    include_visual_insights: bool = True
    detect_contradictions: bool = True
    
    @validator('summary_type')
    def validate_summary_type(cls, v):
        valid_types = [st.value for st in SummaryType]
        if v not in valid_types:
            raise ValueError(f'Summary type must be one of: {valid_types}')
        return v


class ProcessUrlRequest(BaseModel):
    url: HttpUrl
    summary_type: str = "comprehensive"
    max_length: int = 1000
    include_visual_insights: bool = True
    detect_contradictions: bool = True
    
    @validator('url')
    def validate_url(cls, v):
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]


# Global variables for services
redis_client: Optional[redis.Redis] = None
image_analyzer: Optional[ImageAnalyzer] = None
summarizer: Optional[MultiModalSummarizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Multi-modal AI Agent...")
    
    # Initialize Redis
    global redis_client
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize Image Analyzer
    global image_analyzer
    try:
        image_analyzer = ImageAnalyzer(device="auto")
        logger.info("Image Analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Image Analyzer: {e}")
        raise
    
    # Initialize Summarizer
    global summarizer
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not openai_api_key and not anthropic_api_key:
            logger.warning("No LLM API keys provided. Summarization will not work.")
            summarizer = None
        else:
            summarizer = MultiModalSummarizer(
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key
            )
            logger.info("Summarizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Summarizer: {e}")
        raise
    
    logger.info("Multi-modal AI Agent started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-modal AI Agent...")
    
    if redis_client:
        await redis_client.close()
    
    logger.info("Multi-modal AI Agent shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Multi-modal AI Agent",
    description="Advanced multi-modal content summarization service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions
async def get_redis():
    """Get Redis client."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client


async def get_image_analyzer():
    """Get Image Analyzer instance."""
    if not image_analyzer:
        raise HTTPException(status_code=503, detail="Image Analyzer not available")
    return image_analyzer


async def get_summarizer():
    """Get Summarizer instance."""
    if not summarizer:
        raise HTTPException(status_code=503, detail="Summarizer not available")
    return summarizer


# Utility functions
async def get_cache_key(prefix: str, identifier: str) -> str:
    """Generate cache key."""
    return f"{prefix}:{identifier}"


async def get_cached_result(redis_client: redis.Redis, key: str) -> Optional[Dict[str, Any]]:
    """Get cached result from Redis."""
    try:
        cached = await redis_client.get(key)
        if cached:
            return eval(cached)  # In production, use proper JSON serialization
        return None
    except Exception as e:
        logger.warning(f"Error getting cached result: {e}")
        return None


async def cache_result(redis_client: redis.Redis, key: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache result in Redis."""
    try:
        await redis_client.setex(key, ttl, str(result))
    except Exception as e:
        logger.warning(f"Error caching result: {e}")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-modal AI Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {
        "redis": "healthy" if redis_client else "unavailable",
        "image_analyzer": "healthy" if image_analyzer else "unavailable",
        "summarizer": "healthy" if summarizer else "unavailable"
    }
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=time.time(),
        version="1.0.0",
        services=services
    )


@app.post("/scrape")
async def scrape_content(
    request: ScrapeRequest,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Scrape content from a URL.
    
    Extracts text content, images, and metadata from web pages.
    """
    try:
        # Check cache first
        cache_key = await get_cache_key("scrape", str(request.url))
        cached_result = await get_cached_result(redis_client, cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, status_code=200)
        
        logger.info(f"Scraping content from: {request.url}")
        
        # Scrape content
        content = await scrape_article(str(request.url), request.headless)
        
        # Cache result
        await cache_result(redis_client, cache_key, content)
        
        return JSONResponse(content=content, status_code=200)
        
    except Exception as e:
        logger.error(f"Error scraping {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/analyze")
async def analyze_images(
    request: AnalyzeRequest,
    image_analyzer: ImageAnalyzer = Depends(get_image_analyzer),
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Analyze images for content extraction.
    
    Performs OCR, image captioning, object detection, and chart analysis.
    """
    try:
        # Check cache first
        cache_key = await get_cache_key("analyze", str(hash(str(request.image_urls) + request.text_context)))
        cached_result = await get_cached_result(redis_client, cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, status_code=200)
        
        logger.info(f"Analyzing {len(request.image_urls)} images")
        
        # Convert URLs to strings
        image_urls = [str(url) for url in request.image_urls]
        
        # Analyze images
        results = await image_analyzer.analyze_images(image_urls, request.text_context)
        
        # Cache result
        await cache_result(redis_client, cache_key, results)
        
        return JSONResponse(content=results, status_code=200)
        
    except Exception as e:
        logger.error(f"Error analyzing images: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@app.post("/summarize")
async def summarize_content(
    request: SummarizeRequest,
    summarizer: MultiModalSummarizer = Depends(get_summarizer),
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Generate multi-modal summary.
    
    Creates comprehensive summaries combining text and visual content.
    """
    try:
        # Check cache first
        cache_key = await get_cache_key("summarize", str(hash(str(request.text_content) + str(request.image_analysis))))
        cached_result = await get_cached_result(redis_client, cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, status_code=200)
        
        logger.info("Generating multi-modal summary")
        
        # Create summary request
        summary_request = SummaryRequest(
            text_content=request.text_content,
            image_analysis=request.image_analysis,
            summary_type=SummaryType(request.summary_type),
            max_length=request.max_length,
            include_visual_insights=request.include_visual_insights,
            detect_contradictions=request.detect_contradictions
        )
        
        # Generate summary
        result = await summarizer.summarize(summary_request)
        
        # Convert to dict for JSON response
        result_dict = {
            "summary": result.summary,
            "key_points": result.key_points,
            "visual_insights": result.visual_insights,
            "contradictions": result.contradictions,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "token_usage": result.token_usage
        }
        
        # Cache result
        await cache_result(redis_client, cache_key, result_dict)
        
        return JSONResponse(content=result_dict, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/process-url")
async def process_url(
    request: ProcessUrlRequest,
    background_tasks: BackgroundTasks,
    image_analyzer: ImageAnalyzer = Depends(get_image_analyzer),
    summarizer: MultiModalSummarizer = Depends(get_summarizer),
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Complete end-to-end processing of a URL.
    
    This endpoint performs the full pipeline:
    1. Scrape content from URL
    2. Analyze extracted images
    3. Generate multi-modal summary
    """
    try:
        # Check cache first
        cache_key = await get_cache_key("process", str(request.url))
        cached_result = await get_cached_result(redis_client, cache_key)
        if cached_result:
            return JSONResponse(content=cached_result, status_code=200)
        
        logger.info(f"Processing URL: {request.url}")
        start_time = time.time()
        
        # Step 1: Scrape content
        logger.info("Step 1: Scraping content...")
        scraped_content = await scrape_article(str(request.url))
        
        if not scraped_content.get('text_content'):
            raise HTTPException(status_code=400, detail="No text content found")
        
        # Step 2: Analyze images
        logger.info("Step 2: Analyzing images...")
        image_urls = [img['url'] for img in scraped_content.get('images', [])]
        image_analysis = []
        
        if image_urls:
            image_analysis = await image_analyzer.analyze_images(
                image_urls, scraped_content['text_content']
            )
        
        # Step 3: Generate summary
        logger.info("Step 3: Generating summary...")
        summary_request = SummaryRequest(
            text_content=scraped_content['text_content'],
            image_analysis=image_analysis,
            summary_type=SummaryType(request.summary_type),
            max_length=request.max_length,
            include_visual_insights=request.include_visual_insights,
            detect_contradictions=request.detect_contradictions
        )
        
        summary_result = await summarizer.summarize(summary_request)
        
        # Prepare final result
        result = {
            "url": str(request.url),
            "title": scraped_content.get('title', ''),
            "processing_time": time.time() - start_time,
            "scraped_content": {
                "text_length": len(scraped_content.get('text_content', '')),
                "images_found": len(scraped_content.get('images', [])),
                "metadata": scraped_content.get('metadata', {})
            },
            "image_analysis": {
                "total_images": len(image_analysis),
                "successful_analyses": len([img for img in image_analysis if 'error' not in img]),
                "results": image_analysis
            },
            "summary": {
                "text": summary_result.summary,
                "key_points": summary_result.key_points,
                "visual_insights": summary_result.visual_insights,
                "contradictions": summary_result.contradictions,
                "confidence_score": summary_result.confidence_score,
                "model_used": summary_result.model_used,
                "token_usage": summary_result.token_usage
            }
        }
        
        # Cache result
        await cache_result(redis_client, cache_key, result, ttl=7200)  # 2 hours
        
        # Add background task for cleanup
        background_tasks.add_task(cleanup_processing, str(request.url))
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Error processing URL {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/cache/clear")
async def clear_cache(redis_client: redis.Redis = Depends(get_redis)):
    """Clear all cached results."""
    try:
        await redis_client.flushdb()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.get("/cache/stats")
async def cache_stats(redis_client: redis.Redis = Depends(get_redis)):
    """Get cache statistics."""
    try:
        info = await redis_client.info()
        return {
            "total_keys": info.get('db0', {}).get('keys', 0),
            "memory_usage": info.get('used_memory_human', 'N/A'),
            "uptime": info.get('uptime_in_seconds', 0)
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


# Background tasks
async def cleanup_processing(url: str):
    """Cleanup after processing."""
    try:
        logger.info(f"Cleaning up after processing: {url}")
        # Add any cleanup logic here
        await asyncio.sleep(1)  # Simulate cleanup time
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Multi-modal AI Agent API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Multi-modal AI Agent API shutting down...")


# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    # Set environment variables for testing
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key")
    os.environ.setdefault("ANTHROPIC_API_KEY", "your-anthropic-api-key")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

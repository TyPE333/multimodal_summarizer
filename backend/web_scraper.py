"""
Web Scraper Module for Multi-modal Content Extraction

This module provides robust web scraping capabilities for extracting text content,
images, and metadata from web articles and reports. It handles dynamic content
and various web structures with comprehensive error handling.
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse
import logging

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page
from PIL import Image
import io

from loguru import logger


class WebScraper:
    """
    Advanced web scraper for extracting multimodal content from web pages.
    
    Features:
    - Dynamic content handling with Playwright
    - Image extraction and validation
    - Text preprocessing and cleaning
    - Metadata extraction
    - Robust error handling
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize the web scraper.
        
        Args:
            headless: Run browser in headless mode
            timeout: Page load timeout in milliseconds
        """
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Common selectors for content extraction
        self.content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            'main',
            '.main-content'
        ]
        
        # Selectors to exclude
        self.exclude_selectors = [
            'nav',
            'header',
            'footer',
            '.sidebar',
            '.advertisement',
            '.ads',
            '.comments',
            '.social-share',
            '.related-posts'
        ]
        
        # Image selectors
        self.image_selectors = [
            'img[src]',
            'img[data-src]',
            'img[data-lazy-src]',
            'picture img',
            '.image img'
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()
    
    async def start_browser(self):
        """Start the browser instance."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1920, "height": 1080})
            logger.info("Browser started successfully")
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise
    
    async def close_browser(self):
        """Close the browser instance."""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a given URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            logger.info(f"Starting to scrape: {url}")
            
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Load page with retry mechanism
            await self._load_page_with_retry(url)
            
            # Extract content
            content = {
                'url': url,
                'title': await self._extract_title(),
                'text_content': await self._extract_text_content(),
                'images': await self._extract_images(),
                'metadata': await self._extract_metadata(),
                'timestamp': time.time()
            }
            
            # Clean and validate content
            content = self._clean_content(content)
            
            logger.info(f"Successfully scraped {url}: {len(content['text_content'])} chars, {len(content['images'])} images")
            return content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise
    
    async def _load_page_with_retry(self, url: str, max_retries: int = 3):
        """Load page with retry mechanism."""
        for attempt in range(max_retries):
            try:
                await self.page.goto(url, timeout=self.timeout, wait_until='networkidle')
                
                # Wait for dynamic content to load
                await asyncio.sleep(2)
                
                # Scroll to load lazy-loaded content
                await self._scroll_page()
                
                return
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _scroll_page(self):
        """Scroll page to load lazy-loaded content."""
        try:
            # Scroll to bottom
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            
            # Scroll back to top
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Error during page scrolling: {e}")
    
    async def _extract_title(self) -> str:
        """Extract page title."""
        try:
            title = await self.page.title()
            return title.strip() if title else ""
        except Exception as e:
            logger.warning(f"Error extracting title: {e}")
            return ""
    
    async def _extract_text_content(self) -> str:
        """Extract main text content from the page."""
        try:
            # Try to find main content area
            content_text = ""
            
            for selector in self.content_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        content_text = await element.inner_text()
                        if len(content_text) > 100:  # Minimum content length
                            break
                except Exception:
                    continue
            
            # If no main content found, extract from body
            if not content_text or len(content_text) < 100:
                body = await self.page.query_selector('body')
                if body:
                    content_text = await body.inner_text()
            
            return self._clean_text(content_text)
            
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return ""
    
    async def _extract_images(self) -> List[Dict[str, Any]]:
        """Extract images from the page."""
        try:
            images = []
            
            # Get all image elements
            image_elements = await self.page.query_selector_all('img')
            
            for img in image_elements:
                try:
                    # Get image attributes
                    src = await img.get_attribute('src')
                    alt = await img.get_attribute('alt') or ""
                    title = await img.get_attribute('title') or ""
                    
                    # Handle lazy loading
                    if not src:
                        src = await img.get_attribute('data-src') or await img.get_attribute('data-lazy-src')
                    
                    if src:
                        # Make URL absolute
                        src = urljoin(self.page.url, src)
                        
                        # Get image dimensions
                        width = await img.get_attribute('width')
                        height = await img.get_attribute('height')
                        
                        # Validate image
                        if await self._is_valid_image(src):
                            images.append({
                                'url': src,
                                'alt_text': alt,
                                'title': title,
                                'width': int(width) if width else None,
                                'height': int(height) if height else None
                            })
                
                except Exception as e:
                    logger.warning(f"Error processing image: {e}")
                    continue
            
            # Remove duplicates and filter by size
            unique_images = self._deduplicate_images(images)
            filtered_images = self._filter_images_by_size(unique_images)
            
            return filtered_images[:10]  # Limit to 10 images
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []
    
    async def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the page."""
        try:
            metadata = {}
            
            # Extract meta tags
            meta_tags = await self.page.query_selector_all('meta')
            for meta in meta_tags:
                name = await meta.get_attribute('name') or await meta.get_attribute('property')
                content = await meta.get_attribute('content')
                if name and content:
                    metadata[name] = content
            
            # Extract Open Graph tags
            og_tags = {}
            for key, value in metadata.items():
                if key.startswith('og:'):
                    og_tags[key[3:]] = value
            
            if og_tags:
                metadata['open_graph'] = og_tags
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    async def _is_valid_image(self, url: str) -> bool:
        """Check if image URL is valid and accessible."""
        try:
            # Basic URL validation
            if not url or not url.startswith(('http://', 'https://')):
                return False
            
            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
            parsed_url = urlparse(url)
            if not any(parsed_url.path.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Try to fetch image headers
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                return content_type.startswith('image/')
            
            return False
            
        except Exception:
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common boilerplate
        boilerplate_patterns = [
            r'cookie policy',
            r'privacy policy',
            r'terms of service',
            r'subscribe to newsletter',
            r'follow us on',
            r'share this article',
            r'related articles',
            r'comments',
            r'advertisement',
            r'sponsored content'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _deduplicate_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate images based on URL."""
        seen_urls = set()
        unique_images = []
        
        for img in images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        return unique_images
    
    def _filter_images_by_size(self, images: List[Dict[str, Any]], min_width: int = 100, min_height: int = 100) -> List[Dict[str, Any]]:
        """Filter images by minimum dimensions."""
        filtered_images = []
        
        for img in images:
            width = img.get('width')
            height = img.get('height')
            
            # If dimensions are available, filter by size
            if width and height:
                if width >= min_width and height >= min_height:
                    filtered_images.append(img)
            else:
                # If dimensions not available, include the image
                filtered_images.append(img)
        
        return filtered_images
    
    def _clean_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extracted content."""
        # Ensure text content is not too short
        if len(content.get('text_content', '')) < 50:
            logger.warning("Extracted text content is too short")
        
        # Remove images with invalid URLs
        valid_images = []
        for img in content.get('images', []):
            if self._is_valid_url(img['url']):
                valid_images.append(img)
        content['images'] = valid_images
        
        return content
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


async def scrape_article(url: str, headless: bool = True) -> Dict[str, Any]:
    """
    Convenience function to scrape an article.
    
    Args:
        url: The URL to scrape
        headless: Run browser in headless mode
        
    Returns:
        Dictionary containing extracted content
    """
    async with WebScraper(headless=headless) as scraper:
        return await scraper.scrape_url(url)


# Example usage
if __name__ == "__main__":
    async def main():
        url = "https://example.com/article"
        try:
            content = await scrape_article(url)
            print(f"Title: {content['title']}")
            print(f"Text length: {len(content['text_content'])}")
            print(f"Images found: {len(content['images'])}")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())

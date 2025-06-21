"""
Image Analyzer Module for Multi-modal Content Analysis

This module provides comprehensive image analysis capabilities including:
- OCR (Optical Character Recognition) for text extraction
- Image captioning using CLIP and other vision models
- Object detection and classification
- Chart and diagram analysis
- Image preprocessing and validation
"""

import asyncio
import io
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urlparse
import base64

import requests
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import easyocr
from transformers import CLIPProcessor, CLIPModel, pipeline
import torch
from loguru import logger


class ImageAnalyzer:
    """
    Advanced image analyzer for multimodal content processing.
    
    Features:
    - OCR for text extraction from images
    - Image captioning with CLIP
    - Object detection and classification
    - Chart and diagram analysis
    - Image preprocessing and enhancement
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the image analyzer.
        
        Args:
            device: Device to run models on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.ocr_reader = None
        self.clip_processor = None
        self.clip_model = None
        self.object_detector = None
        self.chart_classifier = None
        
        # Initialize models
        self._initialize_models()
        
        # Chart types for classification
        self.chart_types = [
            'bar_chart', 'line_chart', 'pie_chart', 'scatter_plot',
            'histogram', 'heatmap', 'flowchart', 'diagram', 'table'
        ]
        
        logger.info(f"ImageAnalyzer initialized on device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize all required models."""
        try:
            # Initialize OCR
            logger.info("Initializing EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=self.device == 'cuda')
            
            # Initialize CLIP for image captioning
            logger.info("Initializing CLIP model...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Initialize object detection
            logger.info("Initializing object detection...")
            self.object_detector = pipeline(
                "object-detection",
                model="hustvl/yolos-tiny",
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize chart classification
            logger.info("Initializing chart classifier...")
            self.chart_classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def analyze_images(self, image_urls: List[str], text_context: str = "") -> List[Dict[str, Any]]:
        """
        Analyze multiple images asynchronously.
        
        Args:
            image_urls: List of image URLs to analyze
            text_context: Context text from the article
            
        Returns:
            List of analysis results for each image
        """
        try:
            logger.info(f"Starting analysis of {len(image_urls)} images")
            
            # Create tasks for parallel processing
            tasks = [self.analyze_single_image(url, text_context) for url in image_urls]
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing image {image_urls[i]}: {result}")
                    processed_results.append({
                        'url': image_urls[i],
                        'error': str(result),
                        'analysis': {}
                    })
                else:
                    processed_results.append(result)
            
            logger.info(f"Completed analysis of {len(processed_results)} images")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch image analysis: {e}")
            raise
    
    async def analyze_single_image(self, image_url: str, text_context: str = "") -> Dict[str, Any]:
        """
        Analyze a single image comprehensively.
        
        Args:
            image_url: URL of the image to analyze
            text_context: Context text from the article
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            logger.info(f"Analyzing image: {image_url}")
            
            # Download and preprocess image
            image = await self._download_image(image_url)
            if image is None:
                return {
                    'url': image_url,
                    'error': 'Failed to download image',
                    'analysis': {}
                }
            
            # Preprocess image for better analysis
            processed_image = self._preprocess_image(image)
            
            # Perform various analyses
            analysis_results = {
                'url': image_url,
                'image_info': self._get_image_info(image),
                'ocr_text': await self._extract_text(processed_image),
                'caption': await self._generate_caption(processed_image),
                'objects': await self._detect_objects(processed_image),
                'chart_type': await self._classify_chart(processed_image),
                'relevance_score': await self._calculate_relevance(processed_image, text_context),
                'analysis_timestamp': asyncio.get_event_loop().time()
            }
            
            # Generate comprehensive description
            analysis_results['comprehensive_description'] = self._generate_comprehensive_description(
                analysis_results, text_context
            )
            
            logger.info(f"Completed analysis for {image_url}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_url}: {e}")
            return {
                'url': image_url,
                'error': str(e),
                'analysis': {}
            }
    
    async def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"URL does not point to an image: {content_type}")
                return None
            
            # Load image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error downloading image {url}: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better analysis."""
        try:
            # Resize if too large (for performance)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error preprocessing image: {e}")
            return image
    
    def _get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image information."""
        return {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown'),
            'width': image.width,
            'height': image.height
        }
    
    async def _extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text from image using OCR."""
        try:
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Perform OCR
            results = self.ocr_reader.readtext(image_array)
            
            # Process results
            extracted_text = []
            confidence_scores = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    extracted_text.append(text.strip())
                    confidence_scores.append(confidence)
            
            return {
                'text': ' '.join(extracted_text),
                'confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'word_count': len(extracted_text),
                'raw_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_count': 0,
                'raw_results': []
            }
    
    async def _generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generate image caption using CLIP."""
        try:
            # Prepare image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption using CLIP
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
                # Use predefined captions for classification
                caption_templates = [
                    "a photo of a chart showing data",
                    "a diagram illustrating concepts",
                    "a graph displaying statistics",
                    "an image with text and graphics",
                    "a screenshot of a webpage",
                    "a photograph of people or objects",
                    "an illustration or drawing"
                ]
                
                text_inputs = self.clip_processor(text=caption_templates, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features)
                best_match_idx = similarity.argmax().item()
                
                caption = caption_templates[best_match_idx]
                confidence = similarity[best_match_idx].item()
            
            return {
                'caption': caption,
                'confidence': confidence,
                'model': 'CLIP'
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {
                'caption': 'Unable to generate caption',
                'confidence': 0.0,
                'model': 'CLIP'
            }
    
    async def _detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """Detect objects in the image."""
        try:
            # Convert PIL image to format expected by pipeline
            image_array = np.array(image)
            
            # Perform object detection
            results = self.object_detector(image_array)
            
            # Process results
            detected_objects = []
            for result in results:
                detected_objects.append({
                    'label': result['label'],
                    'confidence': result['score'],
                    'bbox': result['box']
                })
            
            return {
                'objects': detected_objects,
                'object_count': len(detected_objects),
                'model': 'YOLOS'
            }
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {
                'objects': [],
                'object_count': 0,
                'model': 'YOLOS'
            }
    
    async def _classify_chart(self, image: Image.Image) -> Dict[str, Any]:
        """Classify if image is a chart and determine its type."""
        try:
            # Convert PIL image to format expected by pipeline
            image_array = np.array(image)
            
            # Perform classification
            results = self.chart_classifier(image_array)
            
            # Analyze results for chart-like characteristics
            chart_indicators = []
            for result in results[:5]:  # Top 5 predictions
                label = result['label'].lower()
                if any(keyword in label for keyword in ['chart', 'graph', 'diagram', 'plot']):
                    chart_indicators.append({
                        'type': label,
                        'confidence': result['score']
                    })
            
            # Determine chart type based on visual analysis
            chart_type = self._analyze_chart_characteristics(image)
            
            return {
                'is_chart': len(chart_indicators) > 0,
                'chart_type': chart_type,
                'confidence': max([r['confidence'] for r in chart_indicators]) if chart_indicators else 0.0,
                'indicators': chart_indicators
            }
            
        except Exception as e:
            logger.error(f"Error in chart classification: {e}")
            return {
                'is_chart': False,
                'chart_type': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
    
    def _analyze_chart_characteristics(self, image: Image.Image) -> str:
        """Analyze image characteristics to determine chart type."""
        try:
            # Convert to numpy array for OpenCV processing
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect lines (for line charts, bar charts)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            # Detect circles (for pie charts)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            
            # Analyze characteristics
            if circles is not None:
                return 'pie_chart'
            elif lines is not None and len(lines) > 10:
                # Count horizontal vs vertical lines
                horizontal_lines = 0
                vertical_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
                        horizontal_lines += 1
                    else:  # Vertical line
                        vertical_lines += 1
                
                if horizontal_lines > vertical_lines:
                    return 'bar_chart'
                else:
                    return 'line_chart'
            else:
                return 'diagram'
                
        except Exception as e:
            logger.warning(f"Error in chart characteristic analysis: {e}")
            return 'unknown'
    
    async def _calculate_relevance(self, image: Image.Image, text_context: str) -> float:
        """Calculate relevance score between image and text context."""
        try:
            if not text_context:
                return 0.5  # Default relevance if no context
            
            # Generate image caption
            caption_result = await self._generate_caption(image)
            image_caption = caption_result['caption']
            
            # Simple keyword matching for relevance
            context_words = set(text_context.lower().split())
            caption_words = set(image_caption.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(context_words.intersection(caption_words))
            union = len(context_words.union(caption_words))
            
            relevance = intersection / union if union > 0 else 0.0
            
            return min(relevance * 2, 1.0)  # Scale up relevance
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.5
    
    def _generate_comprehensive_description(self, analysis: Dict[str, Any], text_context: str) -> str:
        """Generate a comprehensive description of the image analysis."""
        try:
            parts = []
            
            # Add caption
            if analysis.get('caption', {}).get('caption'):
                parts.append(f"Image shows: {analysis['caption']['caption']}")
            
            # Add OCR text if present
            ocr_text = analysis.get('ocr_text', {}).get('text', '')
            if ocr_text:
                parts.append(f"Contains text: {ocr_text[:200]}{'...' if len(ocr_text) > 200 else ''}")
            
            # Add chart information
            chart_info = analysis.get('chart_type', {})
            if chart_info.get('is_chart'):
                parts.append(f"Appears to be a {chart_info.get('chart_type', 'chart')}")
            
            # Add object information
            objects = analysis.get('objects', {}).get('objects', [])
            if objects:
                object_names = [obj['label'] for obj in objects[:3]]  # Top 3 objects
                parts.append(f"Contains: {', '.join(object_names)}")
            
            # Add relevance information
            relevance = analysis.get('relevance_score', 0.5)
            if relevance > 0.7:
                parts.append("Highly relevant to the article content")
            elif relevance > 0.4:
                parts.append("Moderately relevant to the article content")
            else:
                parts.append("May not be directly related to the article content")
            
            return ' '.join(parts)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive description: {e}")
            return "Unable to generate comprehensive description"
    
    def analyze_image_from_base64(self, base64_string: str, text_context: str = "") -> Dict[str, Any]:
        """Analyze image from base64 encoded string."""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess and analyze
            processed_image = self._preprocess_image(image)
            
            # Run analysis synchronously
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.analyze_single_image_sync(processed_image, text_context))
            
        except Exception as e:
            logger.error(f"Error analyzing base64 image: {e}")
            return {
                'error': str(e),
                'analysis': {}
            }
    
    async def analyze_single_image_sync(self, image: Image.Image, text_context: str = "") -> Dict[str, Any]:
        """Synchronous version of single image analysis."""
        try:
            analysis_results = {
                'image_info': self._get_image_info(image),
                'ocr_text': await self._extract_text(image),
                'caption': await self._generate_caption(image),
                'objects': await self._detect_objects(image),
                'chart_type': await self._classify_chart(image),
                'relevance_score': await self._calculate_relevance(image, text_context),
                'analysis_timestamp': asyncio.get_event_loop().time()
            }
            
            analysis_results['comprehensive_description'] = self._generate_comprehensive_description(
                analysis_results, text_context
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in sync image analysis: {e}")
            return {
                'error': str(e),
                'analysis': {}
            }


# Convenience functions
async def analyze_images_batch(image_urls: List[str], text_context: str = "") -> List[Dict[str, Any]]:
    """Analyze multiple images in batch."""
    analyzer = ImageAnalyzer()
    return await analyzer.analyze_images(image_urls, text_context)


def analyze_single_image_sync(image_url: str, text_context: str = "") -> Dict[str, Any]:
    """Analyze a single image synchronously."""
    analyzer = ImageAnalyzer()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(analyzer.analyze_single_image(image_url, text_context))


# Example usage
if __name__ == "__main__":
    async def main():
        # Example image URLs
        image_urls = [
            "https://example.com/chart.png",
            "https://example.com/diagram.jpg"
        ]
        
        text_context = "This article discusses data visualization and chart analysis."
        
        try:
            results = await analyze_images_batch(image_urls, text_context)
            for result in results:
                print(f"Image: {result['url']}")
                print(f"Caption: {result.get('caption', {}).get('caption', 'N/A')}")
                print(f"OCR Text: {result.get('ocr_text', {}).get('text', 'N/A')}")
                print(f"Chart Type: {result.get('chart_type', {}).get('chart_type', 'N/A')}")
                print("---")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())

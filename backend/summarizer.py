"""
Multi-modal Summarizer Module

This module provides advanced summarization capabilities that combine text content,
image analysis, and OCR results to create comprehensive, coherent summaries.
Features include sophisticated prompt engineering, contradiction detection,
and structured output generation.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import openai
import anthropic
from loguru import logger


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    COMPREHENSIVE = "comprehensive"
    EXECUTIVE = "executive"
    BULLET_POINTS = "bullet_points"
    KEY_TAKEAWAYS = "key_takeaways"
    VISUAL_FOCUSED = "visual_focused"


@dataclass
class SummaryRequest:
    """Request structure for summarization."""
    text_content: str
    image_analysis: List[Dict[str, Any]]
    summary_type: SummaryType = SummaryType.COMPREHENSIVE
    max_length: int = 1000
    include_visual_insights: bool = True
    detect_contradictions: bool = True


@dataclass
class SummaryResponse:
    """Response structure for summarization."""
    summary: str
    key_points: List[str]
    visual_insights: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    model_used: str
    token_usage: Optional[Dict[str, int]] = None


class MultiModalSummarizer:
    """
    Advanced multi-modal summarizer that combines text and visual content.
    
    Features:
    - Multi-LLM support (OpenAI, Anthropic)
    - Sophisticated prompt engineering
    - Contradiction detection between text and images
    - Structured output generation
    - Visual insights extraction
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 default_model: str = "gpt-4"):
        """
        Initialize the summarizer.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            default_model: Default model to use
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.default_model = default_model
        
        # Initialize clients
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            
        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.anthropic_client = None
        
        # Validate configuration
        if not self.openai_client and not self.anthropic_client:
            raise ValueError("At least one API key (OpenAI or Anthropic) must be provided")
        
        logger.info(f"MultiModalSummarizer initialized with model: {default_model}")
    
    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        """
        Generate a comprehensive multi-modal summary.
        
        Args:
            request: SummaryRequest containing content and parameters
            
        Returns:
            SummaryResponse with comprehensive results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Starting multi-modal summarization (type: {request.summary_type.value})")
            
            # Prepare content for summarization
            prepared_content = self._prepare_content(request)
            
            # Generate main summary
            summary_result = await self._generate_summary(prepared_content, request)
            
            # Extract key points
            key_points = await self._extract_key_points(summary_result['summary'], request)
            
            # Generate visual insights
            visual_insights = await self._generate_visual_insights(
                request.image_analysis, request.text_content
            )
            
            # Detect contradictions
            contradictions = []
            if request.detect_contradictions:
                contradictions = await self._detect_contradictions(
                    request.text_content, request.image_analysis
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                summary_result, visual_insights, contradictions
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return SummaryResponse(
                summary=summary_result['summary'],
                key_points=key_points,
                visual_insights=visual_insights,
                contradictions=contradictions,
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=summary_result['model'],
                token_usage=summary_result.get('token_usage')
            )
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            raise
    
    def _prepare_content(self, request: SummaryRequest) -> Dict[str, Any]:
        """Prepare and structure content for summarization."""
        # Extract relevant image information
        image_summaries = []
        for img_analysis in request.image_analysis:
            if 'error' not in img_analysis:
                summary = {
                    'url': img_analysis.get('url', ''),
                    'caption': img_analysis.get('caption', {}).get('caption', ''),
                    'ocr_text': img_analysis.get('ocr_text', {}).get('text', ''),
                    'chart_type': img_analysis.get('chart_type', {}).get('chart_type', ''),
                    'relevance_score': img_analysis.get('relevance_score', 0.0),
                    'comprehensive_description': img_analysis.get('comprehensive_description', '')
                }
                image_summaries.append(summary)
        
        # Filter images by relevance
        relevant_images = [img for img in image_summaries if img['relevance_score'] > 0.3]
        
        return {
            'text_content': request.text_content,
            'images': relevant_images,
            'total_images': len(request.image_analysis),
            'relevant_images': len(relevant_images)
        }
    
    async def _generate_summary(self, content: Dict[str, Any], request: SummaryRequest) -> Dict[str, Any]:
        """Generate the main summary using LLM."""
        prompt = self._build_summary_prompt(content, request)
        
        try:
            if self.openai_client and self.default_model.startswith('gpt'):
                return await self._generate_openai_summary(prompt, request)
            elif self.anthropic_client and self.default_model.startswith('claude'):
                return await self._generate_anthropic_summary(prompt, request)
            else:
                # Fallback to OpenAI
                return await self._generate_openai_summary(prompt, request)
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def _build_summary_prompt(self, content: Dict[str, Any], request: SummaryRequest) -> str:
        """Build sophisticated prompt for summarization."""
        
        # Base prompt template
        base_prompt = f"""You are an expert content analyst specializing in multi-modal content summarization. 
Your task is to create a comprehensive summary that integrates information from both text content and visual elements.

CONTENT TO SUMMARIZE:
Text Content: {content['text_content'][:5000]}...

Visual Elements: {len(content['images'])} relevant images found
"""

        # Add image details
        if content['images']:
            base_prompt += "\nIMAGE ANALYSIS:\n"
            for i, img in enumerate(content['images'], 1):
                base_prompt += f"""
Image {i}:
- Caption: {img['caption']}
- OCR Text: {img['ocr_text']}
- Chart Type: {img['chart_type']}
- Relevance Score: {img['relevance_score']:.2f}
- Description: {img['comprehensive_description']}
"""

        # Add summary type specific instructions
        if request.summary_type == SummaryType.COMPREHENSIVE:
            base_prompt += """
INSTRUCTIONS:
Create a comprehensive summary that:
1. Synthesizes the main points from the text content
2. Integrates insights from visual elements (charts, diagrams, images)
3. Highlights how visual content supports or adds to the text narrative
4. Identifies key data points or trends shown in visualizations
5. Provides context for any technical diagrams or charts
6. Maintains a coherent narrative flow

Format the summary as a well-structured paragraph with clear sections.
"""
        elif request.summary_type == SummaryType.EXECUTIVE:
            base_prompt += """
INSTRUCTIONS:
Create an executive summary that:
1. Captures the most critical insights in 2-3 paragraphs
2. Emphasizes key findings and implications
3. Integrates the most relevant visual insights
4. Focuses on actionable takeaways
5. Uses clear, business-friendly language
"""
        elif request.summary_type == SummaryType.BULLET_POINTS:
            base_prompt += """
INSTRUCTIONS:
Create a bullet-point summary that:
1. Lists key points from the text content
2. Includes visual insights as separate bullet points
3. Uses clear, concise language
4. Organizes information logically
5. Highlights data points from charts/diagrams
"""
        elif request.summary_type == SummaryType.KEY_TAKEAWAYS:
            base_prompt += """
INSTRUCTIONS:
Extract the most important takeaways that:
1. Represent the core message of the content
2. Include insights from both text and visual elements
3. Are actionable or informative
4. Can stand alone as key insights
"""
        elif request.summary_type == SummaryType.VISUAL_FOCUSED:
            base_prompt += """
INSTRUCTIONS:
Create a visual-focused summary that:
1. Prioritizes insights from visual elements
2. Explains what each chart/diagram shows
3. Connects visual data to the text narrative
4. Highlights patterns or trends in visualizations
5. Provides context for technical diagrams
"""

        base_prompt += f"""
OUTPUT REQUIREMENTS:
- Maximum length: {request.max_length} words
- Include visual insights: {request.include_visual_insights}
- Focus on accuracy and coherence
- Maintain professional tone
- Ensure all visual elements are properly contextualized

Please provide your summary now:"""

        return base_prompt
    
    async def _generate_openai_summary(self, prompt: str, request: SummaryRequest) -> Dict[str, Any]:
        """Generate summary using OpenAI API."""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are an expert content analyst and summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.max_length * 2,  # Approximate token count
                temperature=0.3,
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                'summary': summary,
                'model': self.default_model,
                'token_usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _generate_anthropic_summary(self, prompt: str, request: SummaryRequest) -> Dict[str, Any]:
        """Generate summary using Anthropic API."""
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=request.max_length * 2,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.content[0].text.strip()
            
            return {
                'summary': summary,
                'model': 'claude-3-sonnet',
                'token_usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _extract_key_points(self, summary: str, request: SummaryRequest) -> List[str]:
        """Extract key points from the summary."""
        try:
            prompt = f"""
Extract 5-7 key points from the following summary. Each point should be:
- Concise (1-2 sentences)
- Actionable or informative
- Representative of a main idea

Summary: {summary}

Return the key points as a numbered list:"""

            if self.openai_client:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.2
                )
                result = response.choices[0].message.content.strip()
            else:
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    temperature=0.2,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.content[0].text.strip()
            
            # Parse numbered list
            key_points = []
            for line in result.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    # Remove numbering/bullets
                    clean_line = re.sub(r'^\d+\.\s*|^[•\-*]\s*', '', line)
                    if clean_line:
                        key_points.append(clean_line)
            
            return key_points[:7]  # Limit to 7 points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []
    
    async def _generate_visual_insights(self, image_analysis: List[Dict[str, Any]], text_content: str) -> List[Dict[str, Any]]:
        """Generate insights specifically from visual content."""
        try:
            insights = []
            
            for img in image_analysis:
                if 'error' in img:
                    continue
                
                # Create insight for each relevant image
                insight = {
                    'image_url': img.get('url', ''),
                    'insight_type': self._determine_insight_type(img),
                    'description': img.get('comprehensive_description', ''),
                    'relevance_score': img.get('relevance_score', 0.0),
                    'data_points': self._extract_data_points(img),
                    'chart_insights': self._extract_chart_insights(img)
                }
                
                if insight['relevance_score'] > 0.3:  # Only include relevant insights
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating visual insights: {e}")
            return []
    
    def _determine_insight_type(self, img_analysis: Dict[str, Any]) -> str:
        """Determine the type of insight from image analysis."""
        chart_type = img_analysis.get('chart_type', {}).get('chart_type', '')
        
        if chart_type in ['bar_chart', 'line_chart', 'pie_chart']:
            return 'data_visualization'
        elif img_analysis.get('ocr_text', {}).get('text', ''):
            return 'text_extraction'
        elif img_analysis.get('objects', {}).get('object_count', 0) > 0:
            return 'object_detection'
        else:
            return 'general_visual'
    
    def _extract_data_points(self, img_analysis: Dict[str, Any]) -> List[str]:
        """Extract data points from OCR text."""
        ocr_text = img_analysis.get('ocr_text', {}).get('text', '')
        if not ocr_text:
            return []
        
        # Look for numbers, percentages, dates
        data_patterns = [
            r'\d+\.?\d*%',  # Percentages
            r'\$\d+\.?\d*',  # Currency
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
            r'\d+\.?\d*',  # General numbers
        ]
        
        data_points = []
        for pattern in data_patterns:
            matches = re.findall(pattern, ocr_text)
            data_points.extend(matches)
        
        return list(set(data_points))  # Remove duplicates
    
    def _extract_chart_insights(self, img_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights specific to charts."""
        chart_info = img_analysis.get('chart_type', {})
        
        if not chart_info.get('is_chart'):
            return {}
        
        return {
            'chart_type': chart_info.get('chart_type', 'unknown'),
            'confidence': chart_info.get('confidence', 0.0),
            'is_data_visualization': chart_info.get('chart_type') in ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot']
        }
    
    async def _detect_contradictions(self, text_content: str, image_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions between text and visual content."""
        try:
            contradictions = []
            
            for img in image_analysis:
                if 'error' in img or img.get('relevance_score', 0) < 0.5:
                    continue
                
                # Analyze potential contradictions
                contradiction = await self._analyze_single_contradiction(text_content, img)
                if contradiction:
                    contradictions.append(contradiction)
            
            return contradictions
            
        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")
            return []
    
    async def _analyze_single_contradiction(self, text_content: str, img_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze potential contradiction for a single image."""
        try:
            prompt = f"""
Analyze if there are any contradictions between the text content and the image description.

Text Content: {text_content[:2000]}...

Image Description: {img_analysis.get('comprehensive_description', '')}
Image Caption: {img_analysis.get('caption', {}).get('caption', '')}
OCR Text: {img_analysis.get('ocr_text', {}).get('text', '')}

Look for:
1. Conflicting data or statistics
2. Contradictory claims or statements
3. Inconsistent information
4. Misleading visual representations

If you find a contradiction, describe it clearly. If no contradiction is found, respond with "No contradiction detected."

Response:"""

            if self.openai_client:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                result = response.choices[0].message.content.strip()
            else:
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model="claude-3-haiku-20240307",
                    max_tokens=300,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.content[0].text.strip()
            
            if "no contradiction detected" not in result.lower():
                return {
                    'image_url': img_analysis.get('url', ''),
                    'contradiction_description': result,
                    'severity': 'medium',  # Could be enhanced with severity classification
                    'confidence': 0.8
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing contradiction: {e}")
            return None
    
    def _calculate_confidence_score(self, summary_result: Dict[str, Any], 
                                  visual_insights: List[Dict[str, Any]], 
                                  contradictions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the summary."""
        try:
            base_score = 0.8  # Base confidence
            
            # Adjust based on summary quality
            summary_length = len(summary_result['summary'])
            if 100 <= summary_length <= 2000:
                base_score += 0.1
            
            # Adjust based on visual insights
            if visual_insights:
                base_score += 0.05
            
            # Penalize for contradictions
            if contradictions:
                base_score -= 0.1 * len(contradictions)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5


# Convenience functions
async def create_summary(text_content: str, 
                        image_analysis: List[Dict[str, Any]], 
                        summary_type: SummaryType = SummaryType.COMPREHENSIVE,
                        openai_api_key: Optional[str] = None,
                        anthropic_api_key: Optional[str] = None) -> SummaryResponse:
    """Convenience function to create a summary."""
    summarizer = MultiModalSummarizer(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key
    )
    
    request = SummaryRequest(
        text_content=text_content,
        image_analysis=image_analysis,
        summary_type=summary_type
    )
    
    return await summarizer.summarize(request)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example usage
        text_content = "This article discusses the impact of AI on healthcare..."
        image_analysis = [
            {
                'url': 'https://example.com/chart.png',
                'caption': {'caption': 'Chart showing AI adoption in healthcare'},
                'ocr_text': {'text': '2023: 45% adoption rate'},
                'chart_type': {'chart_type': 'bar_chart', 'is_chart': True},
                'relevance_score': 0.8,
                'comprehensive_description': 'Bar chart showing healthcare AI adoption rates'
            }
        ]
        
        try:
            summary = await create_summary(
                text_content=text_content,
                image_analysis=image_analysis,
                summary_type=SummaryType.COMPREHENSIVE
            )
            
            print(f"Summary: {summary.summary}")
            print(f"Key Points: {summary.key_points}")
            print(f"Confidence: {summary.confidence_score}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())

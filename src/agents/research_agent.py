"""
Research Agent for etymological data collection.

This module implements the ResearchAgent class, responsible for gathering etymological
data from both local and remote sources. It follows a fallback pattern:
1. Check local vector store
2. Query local etymonline dataset (46,000+ English word etymologies)
3. Query structured APIs (Wiktionary, Etymonline API)
4. Use GPT for analysis
5. Fall back to browser-based scraping
"""

from typing import List, Optional, Tuple
import asyncio
from datetime import datetime
from pathlib import Path
from loguru import logger
from openai import AsyncOpenAI
import os
import json
from dataclasses import dataclass, field
import numpy as np

from src.config import CACHE_DIR
from src.cache.etymology_cache import EtymologyCache
from src.agents.data_sources import (
    LocalDatasetSource,
    WiktionaryAPISource,
    EtymonlineAPISource,
    WebScrapingSource,
    EtymologyDataSource
)
from src.models.etymology_models import EtymologyResult, EtymologySource, LanguageTransition
from src.models.stem_models import Stem, StemAnalysis
from src.agents.similarity_agent import SimilarityAgent
from src.utils.browser import Browser
from src.utils.llm_client import LLMClient

@dataclass
class Stem:
    """
    Represents a morphological stem/root with its properties.
    
    Attributes:
        text: The actual text of the stem
        stem_type: Type of stem (prefix, suffix, root, etc.)
        language: Origin language
        meaning: Definition/meaning of the stem
        etymology: Etymology information
        embedding: Vector embedding for similarity comparison
        similar_roots: List of similar roots with scores
        example_words: Example words using this stem
    """
    text: str
    stem_type: str
    language: str
    meaning: str
    etymology: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    similar_roots: List[Tuple[str, float]] = field(default_factory=list)
    example_words: List[str] = field(default_factory=list)

@dataclass
class StemAnalysis:
    """
    Container for stem analysis results.
    
    Attributes:
        word: The analyzed word
        stems: List of identified stems
        confidence_score: Confidence in the analysis
        notes: Additional analysis notes
    """
    word: str
    stems: List[Stem]
    confidence_score: float
    notes: Optional[str] = None

class ResearchAgent:
    """
    Agent responsible for gathering etymological data from various sources.
    
    This agent follows a hierarchical search pattern:
    1. Check local vector store for existing research
    2. Query local etymonline dataset (46,000+ English word etymologies)
    3. Query structured APIs (Wiktionary, Etymonline API)
    4. Use GPT for analysis
    5. Fall back to browser-based scraping if needed
    """

    def __init__(
        self,
        similarity_agent: Optional[SimilarityAgent] = None,
        browser: Optional[Browser] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the research agent.
        
        Args:
            similarity_agent: Agent for similarity computations
            browser: Browser instance for web scraping
            llm_client: Client for LLM interactions
        """
        self.similarity_agent = similarity_agent
        self.browser = browser
        self.llm_client = llm_client or LLMClient()
        
        # Initialize cache
        self.cache = EtymologyCache()
        
        # Initialize data sources
        self.sources: List[EtymologyDataSource] = [
            LocalDatasetSource(Path("data/etymonline/index.json")),
            WiktionaryAPISource(),
            EtymonlineAPISource(),
            WebScrapingSource()
        ]
        
        # Initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and not api_key.startswith("your_"):
            try:
                self.gpt_client = AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.gpt_client = None
        else:
            self.gpt_client = None
            logger.warning("OpenAI API key not found or invalid. GPT features will be disabled.")
        
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("ResearchAgent initialized")

    async def research_word(self, word: str) -> EtymologyResult:
        """
        Research the etymology of a word using multiple sources.
        
        This method coordinates the research process across multiple data sources,
        merging and validating results. It implements a fallback strategy,
        trying different sources if the primary ones fail.
        
        Args:
            word: Word to research
            
        Returns:
            EtymologyResult containing the research findings
            
        Raises:
            ValueError: If no etymology information can be found
        """
        try:
            word = word.lower().strip()
            logger.info(f"Starting etymology research for word: {word}")
            
            # Check cache first
            cached = await self.cache.get(word)
            if cached:
                logger.info(f"Found cached data for {word}")
                return cached
            
            results = []
            
            # Try etymonline dataset first
            etymonline_result = await self._query_etymonline_dataset(word)
            if etymonline_result:
                logger.info(f"Found {word} in etymonline dataset")
                results.append(etymonline_result)
            
            # Try web scraping if no results yet
            if not results:
                try:
                    async with asyncio.timeout(10):  # 10 second timeout
                        web_result = await self._browser_research(word)
                        if web_result and (web_result.earliest_ancestors or web_result.transitions):
                            results.append(web_result)
                except asyncio.TimeoutError:
                    logger.warning(f"Web Scraping query timed out for {word}")
                except Exception as e:
                    logger.warning(f"Web scraping failed for {word}: {str(e)}")
            
            # Try GPT analysis if available and no results yet
            if not results and self.gpt_client:
                try:
                    async with asyncio.timeout(15):  # 15 second timeout
                        gpt_result = await self._query_gpt(word)
                        if gpt_result and (gpt_result.earliest_ancestors or gpt_result.transitions):
                            results.append(gpt_result)
                except asyncio.TimeoutError:
                    logger.warning(f"GPT analysis timed out for {word}")
                except Exception as e:
                    logger.error(f"GPT analysis failed for {word}: {str(e)}")
            
            # If we have any results, merge them
            if results:
                merged_result = self._merge_results(results)
                if merged_result.earliest_ancestors or merged_result.transitions:
                    await self.cache.store(word, merged_result)
                    return merged_result
            
            # If no results found, return a minimal result
            minimal_result = EtymologyResult(
                word=word,
                earliest_ancestors=[],
                transitions=[],
                sources=[],
                confidence_score=0.0
            )
            return minimal_result
            
        except Exception as e:
            logger.error(f"Research failed for {word}: {str(e)}")
            # Return minimal result instead of raising
            return EtymologyResult(
                word=word,
                earliest_ancestors=[],
                transitions=[],
                sources=[],
                confidence_score=0.0
            )

    async def _query_etymonline_dataset(self, word: str) -> Optional[EtymologyResult]:
        """
        Query the etymonline dataset for the etymology of a word.
        
        Args:
            word: The word to research
            
        Returns:
            EtymologyResult if found, None if not found
        """
        for source in self.sources:
            if isinstance(source, LocalDatasetSource):
                try:
                    result = await source.query(word)
                    if result and (result.earliest_ancestors or result.transitions):
                        return result
                except Exception as e:
                    logger.error(f"Error querying etymonline dataset for {word}: {str(e)}")
        return None

    async def _browser_research(self, word: str) -> Optional[EtymologyResult]:
        """
        Perform browser-based scraping to find etymology information for a word.
        
        Args:
            word: The word to research
            
        Returns:
            EtymologyResult if found, None if not found
        """
        for source in self.sources:
            if isinstance(source, WebScrapingSource):
                result = await source.query(word)
                if result and (result.earliest_ancestors or result.transitions):
                    return result
        return None

    async def _query_gpt(self, word: str) -> Optional[EtymologyResult]:
        """
        Use GPT to analyze etymology based on available knowledge.
        
        Args:
            word: Word to analyze
            
        Returns:
            EtymologyResult if successful, None if analysis fails
        """
        try:
            # Construct prompt
            prompt = f"""Analyze the etymology of the word "{word}" and provide a structured response with:
1. The earliest known ancestors (language and word form)
2. The chronological transitions between languages
3. Approximate dates if known
4. Any relevant notes about semantic or phonetic changes

Format the response as a JSON object with:
{{
    "earliest_ancestors": [["language", "word_form"], ...],
    "transitions": [
        {{
            "from_lang": "source_language",
            "to_lang": "target_language",
            "from_word": "source_word",
            "to_word": "target_word",
            "approximate_date": "date_or_period",
            "notes": "additional_info"
        }},
        ...
    ]
}}"""

            # Query GPT
            response = await self.gpt_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert etymologist specializing in historical linguistics and the evolution of words across languages."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Convert to EtymologyResult
            transitions = [
                LanguageTransition(**trans)
                for trans in result["transitions"]
            ]
            
            return EtymologyResult(
                word=word,
                earliest_ancestors=result["earliest_ancestors"],
                transitions=transitions,
                sources=[EtymologySource(
                    name="GPT Analysis",
                    url="",
                    accessed_date=datetime.utcnow().isoformat(),
                    confidence=0.7  # Lower confidence for AI-generated results
                )],
                confidence_score=0.7
            )
            
        except Exception as e:
            logger.error(f"GPT analysis failed for {word}: {str(e)}")
            return None

    def _merge_results(self, results: List[EtymologyResult]) -> EtymologyResult:
        """
        Merge multiple research results, resolving conflicts and combining information.
        
        Args:
            results: List of research results to merge
            
        Returns:
            Combined EtymologyResult
        """
        all_ancestors = []
        all_transitions = []
        all_sources = []
        confidence_scores = []
        
        # Combine all results
        for result in results:
            all_ancestors.extend(result.earliest_ancestors)
            all_transitions.extend(result.transitions)
            all_sources.extend(result.sources)
            confidence_scores.append(result.confidence_score)
        
        # Remove duplicates while preserving order using a set of tuples
        seen_ancestors = set()
        unique_ancestors = []
        for ancestor in all_ancestors:
            # Convert ancestor tuple to hashable form if needed
            ancestor_key = tuple(str(x) for x in ancestor) if isinstance(ancestor, list) else ancestor
            if ancestor_key not in seen_ancestors:
                seen_ancestors.add(ancestor_key)
                unique_ancestors.append(ancestor)
        
        # Merge similar transitions
        merged_transitions = []
        seen_transitions = set()
        
        for transition in all_transitions:
            key = (transition.from_lang, transition.to_lang, transition.from_word, transition.to_word)
            if key not in seen_transitions:
                merged_transitions.append(transition)
                seen_transitions.add(key)
        
        # Calculate combined confidence score
        final_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.5
        )
        
        return EtymologyResult(
            word=results[0].word,  # All results should be for the same word
            earliest_ancestors=unique_ancestors,
            transitions=merged_transitions,
            sources=all_sources,
            confidence_score=final_confidence
        )

    async def analyze_stems(self, word: str) -> Optional[StemAnalysis]:
        """
        Analyze the morphological stems/roots of a word.
        
        This method breaks down a word into its constituent parts,
        researches their origins and meanings, and computes embeddings
        for similarity analysis.
        
        Args:
            word: Word to analyze
            
        Returns:
            StemAnalysis containing the breakdown and analysis
        """
        try:
            # First get etymology data to help with stem analysis
            etymology_result = await self.research_word(word)
            if not etymology_result:
                return None
            
            # Use LLM to break down word into stems
            stem_prompt = f"""
            Analyze the word '{word}' and break it down into its morphological components.
            Consider prefixes, suffixes, and root words. For each component:
            1. Identify its type (prefix, suffix, root)
            2. Determine its origin language
            3. Provide its meaning/definition
            4. Include etymology information
            5. List example words using this component
            
            Use the etymology data: {etymology_result.transitions}
            """
            
            stem_response = await self.llm_client.analyze_morphology(stem_prompt)
            
            # Process LLM response into Stem objects
            stems = []
            for stem_data in stem_response.stems:
                # Get embedding for the stem
                stem_embedding = await self.similarity_agent.get_embedding(
                    f"{stem_data.text} ({stem_data.meaning})"
                )
                
                # Find similar roots
                similar_roots = await self.similarity_agent.find_similar_roots(
                    stem_embedding,
                    top_k=5,
                    threshold=0.7
                )
                
                stem = Stem(
                    text=stem_data.text,
                    stem_type=stem_data.type,
                    language=stem_data.language,
                    meaning=stem_data.meaning,
                    etymology=stem_data.etymology,
                    embedding=stem_embedding,
                    similar_roots=similar_roots,
                    example_words=stem_data.examples
                )
                stems.append(stem)
            
            return StemAnalysis(
                word=word,
                stems=stems,
                confidence_score=etymology_result.confidence_score,
                notes=stem_response.notes
            )
            
        except Exception as e:
            logger.error(f"Error analyzing stems for {word}: {str(e)}")
            return None 
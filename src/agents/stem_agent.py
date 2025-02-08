"""
Stem Analysis Agent for morphological word analysis.

This module implements the StemAgent class, responsible for breaking down
words into their morphological components and analyzing their origins.
"""

import time
import json
from typing import List, Optional, Dict
from loguru import logger

from src.models.stem_models import Stem, StemAnalysis
from src.agents.similarity_agent import SimilarityAgent
from src.agents.research_agent import ResearchAgent
from src.models.etymology_models import EtymologyResult
from src.utils.llm_client import LLMClient
from src.rag import RAGPipeline
from src.utils.logging_config import get_logger, log_method_call, log_method_result

# Get module logger
logger = get_logger(__name__)

class StemAgent:
    """
    Agent responsible for morphological analysis of words.
    
    This agent breaks down words into their constituent stems (roots, prefixes,
    suffixes) and analyzes their origins, meanings, and relationships.
    """
    
    def __init__(
        self,
        research_agent: ResearchAgent,
        similarity_agent: SimilarityAgent,
        rag_pipeline: RAGPipeline,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the stem agent.
        
        Args:
            research_agent: Agent for etymology research
            similarity_agent: Agent for similarity computations
            rag_pipeline: RAG pipeline for data storage/retrieval
            llm_client: Client for LLM interactions
        """
        self.research_agent = research_agent
        self.similarity_agent = similarity_agent
        self.rag_pipeline = rag_pipeline
        self.llm_client = llm_client or LLMClient()
        logger.info("StemAgent initialized with components", 
                   research_agent=type(research_agent).__name__,
                   similarity_agent=type(similarity_agent).__name__,
                   rag_pipeline=type(rag_pipeline).__name__,
                   llm_client=type(llm_client).__name__)
    
    def _create_stem_prompt(self, word: str, etymology: EtymologyResult) -> str:
        """
        Create prompt for stem analysis.
        
        Args:
            word: Word to analyze
            etymology: Etymology data to help analysis
            
        Returns:
            Analysis prompt
        """
        log_method_call("_create_stem_prompt", word=word, etymology_id=id(etymology))
        
        # Extract relevant etymology info
        transitions = []
        for t in etymology.transitions:
            transitions.append(
                f"- {t.from_word} ({t.from_lang}) → {t.to_word} ({t.to_lang})"
            )
        
        # Build detailed prompt with explicit example
        prompt = f"""Analyze the morphological structure of '{word}'.

Etymology context:
{chr(10).join(transitions)}

Break down the word into its morphological components (prefixes, roots, suffixes).
Return a valid JSON object with an array of stems, where each stem has:
- text: The stem text
- stem_type: "prefix", "root", or "suffix"
- position: [start_idx, end_idx] in the word
- language: Origin language details
- meaning: Core and extended meanings
- etymology: Historical development
- morphology: Allomorphs and combinations
- examples: Usage examples
- confidence: 0.0-1.0 score

For example, analyzing "dichotomy" should return stems like:
{
  "stems": [
    {
      "text": "di-",
      "stem_type": "prefix",
      "position": [0, 2],
      "language": {"origin": "Greek", "family": "Indo-European"},
      "meaning": {"core": "two", "extended": ["double", "twice"]},
      "etymology": {"development": "From Greek di-"},
      "morphology": {"allomorphs": ["dy-"], "combinations": ["di-sect"]},
      "examples": {"modern": ["divide", "diverge"], "historical": ["dicho-"]},
      "confidence": 0.9
    },
    {
      "text": "-chot-",
      "stem_type": "root",
      "position": [2, 6],
      "language": {"origin": "Greek", "family": "Indo-European"},
      "meaning": {"core": "cut", "extended": ["divide", "split"]},
      "etymology": {"development": "From Greek khot-"},
      "morphology": {"allomorphs": ["tom-"], "combinations": ["anatomy"]},
      "examples": {"modern": ["tome", "atom"], "historical": ["tomos"]},
      "confidence": 0.8
    },
    {
      "text": "-omy",
      "stem_type": "suffix",
      "position": [6, 9],
      "language": {"origin": "Greek", "family": "Indo-European"},
      "meaning": {"core": "process", "extended": ["study of", "cutting"]},
      "etymology": {"development": "From Greek -omia"},
      "morphology": {"allomorphs": ["-tomy"], "combinations": ["anatomy"]},
      "examples": {"modern": ["anatomy", "economy"], "historical": ["tomia"]},
      "confidence": 0.9
    }
  ]
}

Consider:
1. Common prefixes/suffixes in the word's origin language
2. Morphological patterns in related words
3. Historical sound changes
4. Semantic development
5. Word formation rules

Provide detailed analysis with high confidence for clear components,
and lower confidence scores for inferred elements."""

        logger.debug("Created stem analysis prompt", 
                    word=word,
                    transitions_count=len(transitions),
                    prompt_length=len(prompt))
        return prompt
    
    async def analyze_word(self, word: str) -> Optional[StemAnalysis]:
        """
        Analyze the morphological structure of a word.
        
        Args:
            word: The word to analyze
            
        Returns:
            StemAnalysis object if successful, None otherwise
        """
        try:
            logger.info(f"Starting morphological analysis for word: {word}")
            
            # First check cache
            cached = await self.rag_pipeline.get_stem_analysis(word)
            if cached:
                logger.info(f"Found cached stem analysis for {word}")
                logger.debug(f"Cached analysis: {cached}")
                return cached
            
            logger.info(f"No cached analysis found for {word}, generating new analysis")
            
            # Create the analysis prompt
            prompt = self._create_stem_prompt(word)
            logger.debug(f"Generated analysis prompt: {prompt}")
            
            # Get LLM response
            try:
                response = await self.llm_client.analyze_morphology(prompt)
                logger.debug(f"Raw LLM response: {json.dumps(response, indent=2)}")
            except Exception as e:
                logger.error(f"Error getting LLM response: {str(e)}")
                return None
            
            if not response:
                logger.warning(f"No valid response from LLM for {word}")
                return None
            
            # Create StemAnalysis object
            try:
                analysis = StemAnalysis(
                    word=word,
                    stems=response.stems,
                    confidence_score=response.confidence_score,
                    notes=response.notes,
                    metadata={"source": "llm"}
                )
                logger.info(f"Successfully created stem analysis for {word}")
                logger.debug(f"Final analysis: {analysis}")
                
                # Cache the result
                await self.rag_pipeline.store_stem_analysis(analysis)
                logger.info(f"Cached stem analysis for {word}")
                
                return analysis
                
            except Exception as e:
                logger.error(f"Error creating StemAnalysis object: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error in analyze_word for {word}: {str(e)}")
            return None

    async def analyze_word(self, word: str) -> Optional[StemAnalysis]:
        """
        Analyze the morphological structure of a word.
        
        Args:
            word: Word to analyze
            
        Returns:
            StemAnalysis containing the breakdown and analysis
        """
        start_time = time.time()
        log_method_call("analyze_word", word=word)
        
        try:
            # First check cache
            cached = await self.rag_pipeline.get_stem_analysis(word)
            if cached:
                logger.info(f"Found cached stem analysis for {word}")
                duration = time.time() - start_time
                log_method_result("analyze_word", "cached_result", duration)
                return cached
            
            # Get etymology data to help with analysis
            etymology = await self.research_agent.research_word(word)
            if not etymology:
                logger.warning(f"No etymology data found for {word}")
                duration = time.time() - start_time
                log_method_result("analyze_word", None, duration)
                return None
            
            # Use LLM to break down word into stems
            try:
                stem_prompt = self._create_stem_prompt(word, etymology)
                logger.debug("Sending stem analysis prompt to LLM", word=word, prompt_length=len(stem_prompt))
                stem_response = await self.llm_client.analyze_morphology(stem_prompt)
                if not stem_response or not stem_response.stems:
                    logger.warning(f"No stems found in LLM response for {word}")
                    duration = time.time() - start_time
                    log_method_result("analyze_word", None, duration)
                    return None
            except Exception as e:
                logger.error(f"LLM analysis failed for {word}: {str(e)}")
                duration = time.time() - start_time
                log_method_result("analyze_word", None, duration)
                return None
            
            # Process stems with enhanced analysis
            stems = []
            for stem_data in stem_response.stems:
                try:
                    # Generate rich embedding combining multiple aspects
                    embedding_text = (
                        f"{stem_data['text']} "
                        f"{stem_data['meaning']['core']} "
                        f"{' '.join(stem_data['meaning'].get('extended', []))} "
                        f"{' '.join(stem_data['etymology'].get('cognates', []))}"
                    )
                    logger.debug("Generating embedding for stem", 
                               stem=stem_data['text'],
                               text_length=len(embedding_text))
                    stem_embedding = await self.similarity_agent.get_embedding(embedding_text)
                    
                    # Find similar roots with expanded context
                    similar_roots = await self.similarity_agent.find_similar_roots(
                        stem_embedding,
                        top_k=10,
                        threshold=0.65
                    )
                    logger.debug("Found similar roots", 
                               stem=stem_data['text'],
                               similar_count=len(similar_roots))
                    
                    # Create enhanced stem object
                    stem = Stem(
                        text=stem_data['text'],
                        stem_type=stem_data.get('stem_type', stem_data.get('type')),
                        position=tuple(stem_data['position']),
                        language=stem_data['language'],
                        meaning=stem_data['meaning'],
                        etymology=stem_data['etymology'],
                        morphology=stem_data['morphology'],
                        examples=stem_data['examples'],
                        confidence=stem_data.get('confidence', 0.8)
                    )
                    stems.append(stem)
                    logger.info(f"Processed stem: {stem.text} ({stem.stem_type})")
                except Exception as e:
                    logger.error(f"Error processing stem {stem_data.get('text', 'unknown')}: {str(e)}")
                    continue
            
            if not stems:
                logger.warning(f"No valid stems found for {word}")
                duration = time.time() - start_time
                log_method_result("analyze_word", None, duration)
                return None
            
            # Create enhanced analysis result
            result = StemAnalysis(
                word=word,
                stems=stems,
                confidence_score=etymology.confidence_score,
                notes=stem_response.notes,
                metadata={
                    "etymology_sources": [s.name for s in etymology.sources],
                    "analysis_date": stem_response.timestamp,
                    "word_formation_pattern": stem_response.word_formation_pattern,
                    "morphological_features": stem_response.morphological_features,
                    "development_insights": stem_response.development_insights,
                    "usage_patterns": stem_response.usage_patterns
                }
            )
            
            # Cache result
            try:
                await self.rag_pipeline.store_stem_analysis(result)
                logger.info(f"Cached stem analysis for {word}")
            except Exception as e:
                logger.error(f"Failed to cache stem analysis for {word}: {str(e)}")
            
            duration = time.time() - start_time
            log_method_result("analyze_word", f"success ({len(stems)} stems)", duration)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing stems for {word}: {str(e)}")
            duration = time.time() - start_time
            log_method_result("analyze_word", f"error: {str(e)}", duration)
            return None 
"""
Main orchestration module for the Etymologistics application.

This module coordinates the agent pipeline, RAG system, and GUI components.
It's designed for easy maintenance and extensibility, with clear separation
of concerns and minimal coupling between components.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import sys
import asyncio
import streamlit as st

from src.agents.research_agent import ResearchAgent, EtymologyResult
from src.agents.tree_agent import TreeAgent, EtymologyNode
from src.agents.similarity_agent import SimilarityAgent, SimilarityResult
from src.agents.stem_agent import StemAgent
from src.agents.data_sources import LanguageTransition
from src.models.stem_models import StemAnalysis
from src.rag import RAGPipeline, EtymologyDocument
from src.gui import EtymologyUI
from src.config import VECTOR_DB_DIR
from src.utils.browser import Browser
from src.utils.llm_client import LLMClient


@dataclass
class PipelineResult:
    """
    Container for results from the complete pipeline.
    
    This class aggregates results from all agents into a single,
    well-structured object for easier handling and display.
    
    Attributes:
        word: The researched word
        etymology: Research results from etymology sources
        tree: Constructed etymology tree
        similarities: Word similarity analysis
        stem_analysis: Stem analysis results
        metadata: Additional processing information
    """
    word: str
    etymology: EtymologyResult
    tree: EtymologyNode
    similarities: SimilarityResult
    stem_analysis: Optional[StemAnalysis] = None
    metadata: Dict = None

    @property
    def transitions(self) -> List[LanguageTransition]:
        """Get transitions from the etymology result."""
        return self.etymology.transitions if self.etymology else []

    @property
    def earliest_ancestors(self) -> List[Tuple[str, str]]:
        """Get earliest ancestors from the etymology result."""
        return self.etymology.earliest_ancestors if self.etymology else []

    @property
    def confidence_score(self) -> float:
        """Get confidence score from the etymology result."""
        return self.etymology.confidence_score if self.etymology else 0.0
        
    @property
    def relations(self) -> Dict[str, List[Dict]]:
        """Get relationships from the tree data."""
        if not self.tree:
            return {}
        return self.tree.relations if hasattr(self.tree, 'relations') else {}


class EtymologyPipeline:
    """
    Main pipeline coordinator for etymology processing.
    
    This class manages the interaction between agents, ensuring proper
    data flow and error handling. It's designed to be easily extended
    with new agents or processing steps.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        enable_cache: bool = True
    ):
        """
        Initialize the pipeline with all necessary components.
        
        Args:
            data_dir: Directory for data storage
            enable_cache: Whether to use RAG caching
        """
        self.data_dir = data_dir or VECTOR_DB_DIR
        self.enable_cache = enable_cache
        
        # Configure logging
        self._setup_logging()
        
        # Initialize agents
        logger.info("Initializing etymology pipeline...")
        
        # Initialize RAG pipeline if caching enabled
        self.rag_pipeline = RAGPipeline() if enable_cache else None
        
        # Initialize similarity agent
        self.similarity_agent = SimilarityAgent()
        
        # Initialize research agent with dependencies
        self.research_agent = ResearchAgent(
            similarity_agent=self.similarity_agent,
            browser=None,  # Optional, can be added if needed
            llm_client=None  # Will create its own instance
        )
        
        # Initialize tree agent
        self.tree_agent = TreeAgent()
        
        # Initialize stem agent with dependencies
        self.stem_agent = StemAgent(
            research_agent=self.research_agent,
            similarity_agent=self.similarity_agent,
            rag_pipeline=self.rag_pipeline
        )
        
        logger.info("Pipeline initialization complete")
    
    def _setup_logging(self):
        """Configure logging for the application."""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        logger.add(
            self.data_dir / "etymology.log",
            rotation="500 MB",
            retention="10 days",
            level="INFO"
        )
    
    async def process_word(
        self,
        word: str,
        similarity_threshold: float = 0.7
    ) -> PipelineResult:
        """
        Process a word through the complete etymology pipeline.
        
        This method orchestrates the flow between agents, handling
        caching and error recovery. It's the main entry point for
        word processing.
        
        Args:
            word: Word to research
            similarity_threshold: Minimum similarity score
            
        Returns:
            PipelineResult containing all analysis data
            
        Raises:
            ValueError: If word is invalid
            RuntimeError: If pipeline processing fails
        """
        start_time = datetime.now()
        word = word.lower().strip()
        
        if not word:
            raise ValueError("Word cannot be empty")
        
        logger.info(f"Starting pipeline processing for word: {word}")
        
        try:
            # Check cache first
            if self.enable_cache:
                cached_result = await self._check_cache(word)
                if cached_result:
                    logger.info(f"Using cached results for {word}")
                    return cached_result
            
            # Run research pipeline
            result = await self._run_pipeline(word, similarity_threshold)
            
            # Cache results if enabled
            if self.enable_cache:
                await self._cache_results(result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed pipeline for {word} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error processing {word}: {str(e)}")
            raise RuntimeError(f"Failed to process word: {str(e)}")
    
    async def _run_pipeline(
        self,
        word: str,
        similarity_threshold: float
    ) -> PipelineResult:
        """
        Run the core pipeline sequence with concurrent processing.
        
        This method defines the order of operations and data flow
        between agents, executing tasks concurrently where possible.
        
        Args:
            word: Word to process
            similarity_threshold: Minimum similarity score
            
        Returns:
            PipelineResult containing all analysis data
        """
        # Step 1: Etymology Research (required first)
        etymology_result = await self.research_agent.research_word(word)
        logger.debug(f"Completed etymology research for {word}")
        
        # Step 2: Concurrent processing of tree, similarities, and stems
        async def build_tree():
            tree = self.tree_agent.construct_tree(etymology_result)
            logger.debug(f"Constructed etymology tree for {word}")
            return tree
            
        async def compute_similarities():
            reference_words = self._extract_reference_words(etymology_result)
            similarities = await self._compute_similarities(
                word,
                reference_words,
                similarity_threshold
            )
            logger.debug(f"Computed similarities for {word}")
            return similarities
            
        async def analyze_stems():
            stem_analysis = await self.stem_agent.analyze_word(word)
            logger.debug(f"Completed stem analysis for {word}")
            return stem_analysis
            
        # Execute tasks concurrently
        tree_task = asyncio.create_task(build_tree())
        similarities_task = asyncio.create_task(compute_similarities())
        stem_task = asyncio.create_task(analyze_stems())
        
        # Wait for all tasks to complete
        tree, similarities, stem_analysis = await asyncio.gather(
            tree_task, similarities_task, stem_task,
            return_exceptions=True
        )
        
        # Handle any exceptions from concurrent tasks
        for result in [tree, similarities, stem_analysis]:
            if isinstance(result, Exception):
                logger.error(f"Error in concurrent processing: {str(result)}")
                raise result
                
        return PipelineResult(
            word=word,
            etymology=etymology_result,
            tree=tree,
            similarities=similarities,
            stem_analysis=stem_analysis,
            metadata={
                "processing_date": datetime.now().isoformat(),
                "similarity_threshold": similarity_threshold
            }
        )
    
    def _extract_reference_words(
        self,
        etymology_result: EtymologyResult
    ) -> List[Tuple[str, str]]:
        """Extract word-language pairs for similarity analysis."""
        reference_words = []
        
        # Add transitions
        for transition in etymology_result.transitions:
            if transition.from_word and transition.from_lang:
                reference_words.append((transition.from_word, transition.from_lang))
            if transition.to_word and transition.to_word != etymology_result.word and transition.to_lang:
                reference_words.append((transition.to_word, transition.to_lang))
        
        # Add earliest ancestors
        for lang, word_form in etymology_result.earliest_ancestors:
            if lang and word_form:
                reference_words.append((word_form, lang))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_reference_words = []
        for pair in reference_words:
            if pair not in seen:
                seen.add(pair)
                unique_reference_words.append(pair)
        
        return unique_reference_words
    
    async def _compute_similarities(
        self,
        word: str,
        reference_words: List[Tuple[str, str]],
        threshold: float
    ) -> SimilarityResult:
        """Compute similarities with reference words."""
        if not reference_words:
            return SimilarityResult(
                word=word,
                similar_words=[],
                model_name=self.similarity_agent.model_name,
                metric=self.similarity_agent.default_metric,
                metadata={"error": "No reference words available"}
            )
        
        return await self.similarity_agent.compute_similarities(
            word,
            reference_words,
            min_score=threshold
        )
    
    async def _check_cache(self, word: str) -> Optional[PipelineResult]:
        """Check if word results are cached in RAG pipeline."""
        if not self.rag_pipeline:
            return None
            
        try:
            result = await self.rag_pipeline.get_etymology(word)
            if result and result.document:
                return self._convert_cache_to_result(result.document)
        except Exception as e:
            logger.warning(f"Cache check failed for {word}: {str(e)}")
        
        return None
    
    async def _cache_results(self, result: PipelineResult) -> None:
        """Cache pipeline results in RAG system."""
        if not self.rag_pipeline:
            return
            
        try:
            await self.rag_pipeline.store_etymology(
                result.etymology,
                embeddings={"word_embedding": result.similarities}
            )
        except Exception as e:
            logger.warning(f"Failed to cache results for {result.word}: {str(e)}")
    
    def _convert_cache_to_result(
        self,
        document: EtymologyDocument
    ) -> PipelineResult:
        """Convert cached document to pipeline result format."""
        return PipelineResult(
            word=document.word,
            etymology=EtymologyResult(**document.etymology_data),
            tree=self.tree_agent.construct_tree(
                EtymologyResult(**document.etymology_data)
            ),
            similarities=document.embeddings.get("word_embedding"),
            stem_analysis=document.stem_analysis,
            metadata=document.metadata
        )

    async def analyze_stems(self, word: str) -> Optional[StemAnalysis]:
        """
        Analyze the morphological stems/roots of a word.
        
        This method delegates to the stem agent for analysis while handling
        caching and error recovery.
        
        Args:
            word: Word to analyze
            
        Returns:
            StemAnalysis containing the breakdown and analysis
            
        Raises:
            ValueError: If word is invalid
        """
        if not word or not word.strip():
            raise ValueError("Word cannot be empty")
            
        word = word.lower().strip()
        logger.info(f"Starting stem analysis for {word}")
        
        try:
            # Check cache first if enabled
            if self.enable_cache and self.rag_pipeline:
                cached = await self.rag_pipeline.get_stem_analysis(word)
                if cached:
                    logger.info(f"Using cached stem analysis for {word}")
                    return cached
            
            # Perform new analysis
            result = await self.stem_agent.analyze_word(word)
            
            # Cache result if successful
            if result and self.enable_cache and self.rag_pipeline:
                try:
                    await self.rag_pipeline.store_stem_analysis(result)
                except Exception as e:
                    logger.warning(f"Failed to cache stem analysis for {word}: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing stems for {word}: {str(e)}")
            return None


async def main():
    """Initialize and run the application."""
    try:
        # Initialize components
        logger.info("Initializing application components...")
        
        # Initialize browser and LLM client first
        browser = Browser()
        llm_client = LLMClient()
        
        # Initialize agents with dependencies
        similarity_agent = SimilarityAgent()
        research_agent = ResearchAgent(
            similarity_agent=similarity_agent,
            browser=browser,
            llm_client=llm_client
        )
        tree_agent = TreeAgent()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Initialize stem agent with all required dependencies
        stem_agent = StemAgent(
            research_agent=research_agent,
            similarity_agent=similarity_agent,
            rag_pipeline=rag_pipeline
        )
        
        # Initialize pipeline
        pipeline = EtymologyPipeline(
            data_dir=VECTOR_DB_DIR,
            enable_cache=True
        )
        
        # Initialize UI
        ui = EtymologyUI(pipeline=pipeline)
        
        # Run UI
        ui.run()
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
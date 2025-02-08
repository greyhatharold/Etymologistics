"""
Similarity Scoring Agent for etymological analysis.

This module implements the SimilarityAgent class, responsible for computing
similarity scores between words using various embedding and vectorization
techniques. It supports multiple similarity metrics and embedding models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from src.config import VECTOR_DB_CONFIG


@dataclass
class SimilarityResult:
    """
    Represents similarity computation results for a word.
    
    Attributes:
        word: The query word
        similar_words: List of (word, language, score) tuples
        model_name: Name of the embedding model used
        metric: Similarity metric used (e.g., "cosine")
        metadata: Additional computation details
    """
    word: str
    similar_words: List[Tuple[str, str, float]]  # (word, language, score)
    model_name: str
    metric: str
    metadata: Dict = None


class SimilarityAgent:
    """
    Agent responsible for computing word similarities using embeddings.
    
    This agent uses transformer-based models to encode words and compute
    similarity scores. It supports multiple languages and similarity metrics.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the similarity agent with specified model and configuration.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run computations on ("cpu" or "cuda")
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Initialize the embedding model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.to(self.device)
            logger.info(f"Initialized {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
        
        # Set default similarity metric
        self.default_metric = "cosine"
        
        # Initialize embedding cache if directory provided
        self._embedding_cache = {}
        if cache_dir:
            self._load_embedding_cache()
        
        self.rag_pipeline = None  # Will be set by the pipeline

    async def compute_similarities(
        self,
        word: str,
        reference_words: List[Tuple[str, str]],  # (word, language) pairs
        metric: str = None,
        min_score: float = 0.5
    ) -> SimilarityResult:
        """
        Compute similarity scores between a word and reference words.
        
        Args:
            word: Query word to compare
            reference_words: List of (word, language) pairs to compare against
            metric: Similarity metric to use (defaults to cosine)
            min_score: Minimum similarity score to include in results
            
        Returns:
            SimilarityResult containing scored matches
            
        Raises:
            ValueError: If word is empty or invalid
        """
        if not word or not word.strip():
            raise ValueError("Word cannot be empty")
        
        metric = metric or self.default_metric
        word = word.lower().strip()
        
        # Get embeddings
        query_embedding = await self._get_embedding(word)
        ref_embeddings = await self._get_embeddings([w for w, _ in reference_words])
        
        # Compute similarities
        scores = self._compute_similarity(
            query_embedding,
            ref_embeddings,
            metric=metric
        )
        
        # Filter and sort results
        similar_words = []
        for (ref_word, language), score in zip(reference_words, scores):
            if score >= min_score:
                similar_words.append((ref_word, language, float(score)))
        
        similar_words.sort(key=lambda x: x[2], reverse=True)
        
        return SimilarityResult(
            word=word,
            similar_words=similar_words,
            model_name=self.model_name,
            metric=metric,
            metadata={
                "min_score": min_score,
                "total_comparisons": len(reference_words)
            }
        )

    async def _get_embedding(self, word: str) -> NDArray:
        """
        Get embedding for a single word, using cache if available.
        
        Args:
            word: Word to embed
            
        Returns:
            Numpy array containing word embedding
        """
        if word in self._embedding_cache:
            return self._embedding_cache[word]
        
        embedding = self.model.encode(
            [word],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        
        if self.cache_dir:
            self._embedding_cache[word] = embedding
            self._save_embedding(word, embedding)
        
        return embedding

    async def _get_embeddings(self, words: List[str]) -> NDArray:
        """
        Get embeddings for multiple words in batch.
        
        Args:
            words: List of words to embed
            
        Returns:
            Numpy array of embeddings
        """
        # Check cache first
        uncached_words = []
        cached_embeddings = []
        
        for word in words:
            if word in self._embedding_cache:
                cached_embeddings.append(self._embedding_cache[word])
            else:
                uncached_words.append(word)
        
        if uncached_words:
            # Compute new embeddings in batch
            new_embeddings = self.model.encode(
                uncached_words,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Update cache
            if self.cache_dir:
                for word, embedding in zip(uncached_words, new_embeddings):
                    self._embedding_cache[word] = embedding
                    self._save_embedding(word, embedding)
            
            # Combine cached and new embeddings
            all_embeddings = np.vstack(cached_embeddings + [new_embeddings])
        else:
            all_embeddings = np.vstack(cached_embeddings)
        
        return all_embeddings

    def _compute_similarity(
        self,
        query_embedding: NDArray,
        reference_embeddings: NDArray,
        metric: str = "cosine"
    ) -> NDArray:
        """
        Compute similarity scores between query and reference embeddings.
        
        Args:
            query_embedding: Embedding of query word
            reference_embeddings: Matrix of reference word embeddings
            metric: Similarity metric to use
            
        Returns:
            Array of similarity scores
            
        Raises:
            ValueError: If metric is not supported
        """
        if metric == "cosine":
            return cosine_similarity(
                query_embedding.reshape(1, -1),
                reference_embeddings
            )[0]
        elif metric == "dot":
            return np.dot(reference_embeddings, query_embedding)
        elif metric == "euclidean":
            return 1 / (1 + np.linalg.norm(
                reference_embeddings - query_embedding,
                axis=1
            ))
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    def _save_embedding(self, word: str, embedding: NDArray) -> None:
        """Save word embedding to cache directory."""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"{word}.npy"
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding for {word}: {str(e)}")

    def _load_embedding_cache(self) -> None:
        """Load cached embeddings from disk."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.npy"):
                word = cache_file.stem
                self._embedding_cache[word] = np.load(cache_file)
            
            logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
        except Exception as e:
            logger.error(f"Error loading embedding cache: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the embedding cache from memory and disk."""
        self._embedding_cache.clear()
        
        if self.cache_dir and self.cache_dir.exists():
            try:
                for cache_file in self.cache_dir.glob("*.npy"):
                    cache_file.unlink()
                logger.info("Cleared embedding cache")
            except Exception as e:
                logger.error(f"Error clearing cache files: {str(e)}")

    async def compute_root_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute similarity between two root embeddings.
        
        Args:
            embedding1: First root embedding
            embedding2: Second root embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Ensure embeddings are normalized
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Ensure result is between 0 and 1
        return float(max(0.0, min(1.0, similarity)))
    
    async def find_similar_roots(
        self,
        stem_text: str,
        embedding: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar roots to the given stem text.
        
        Args:
            stem_text: Stem text to compare
            embedding: Optional embedding to compare against
            
        Returns:
            List of (root, score) tuples
        """
        try:
            if self.rag_pipeline is None:
                logger.warning("RAG pipeline not initialized")
                return []
                
            if embedding is None:
                # Generate embedding if not provided
                embedding = self.model.encode(stem_text, convert_to_numpy=True)
            
            # Use the RAG pipeline to find similar stems
            similar_stems = await self.rag_pipeline.find_similar_stems(
                embedding=embedding,
                top_k=5,
                threshold=0.7
            )
            
            return similar_stems
            
        except Exception as e:
            logger.error(f"Error finding similar roots: {str(e)}")
            return []
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using the sentence transformer.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Get embedding from model
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,  # Get numpy array
                normalize_embeddings=True  # L2 normalize
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding for '{text}': {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())

    def set_rag_pipeline(self, pipeline) -> None:
        """Set the RAG pipeline reference."""
        self.rag_pipeline = pipeline 
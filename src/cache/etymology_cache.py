"""
Cache implementation for etymology data.

This module provides caching functionality for etymology research results,
using ChromaDB as the underlying storage system.
"""

from pathlib import Path
from typing import Optional
from loguru import logger
import chromadb

from src.models.etymology_models import EtymologyResult, LanguageTransition, EtymologySource, WordRelation
from src.config import VECTOR_DB_CONFIG, VECTOR_DB_DIR


class EtymologyCache:
    """
    Cache for etymology research results.
    
    This class provides a persistent cache for etymology data using ChromaDB
    as the storage backend. It handles serialization and deserialization of
    research results.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the etymology cache.
        
        Args:
            cache_dir: Directory for cache storage (defaults to VECTOR_DB_DIR)
        """
        self.cache_dir = cache_dir or VECTOR_DB_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.cache_dir))
        
        # Get collection name from config with fallback
        collection_name = VECTOR_DB_CONFIG.get("etymology_collection", "etymologies")
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": VECTOR_DB_CONFIG.get("similarity_metric", "cosine"),
                "embedding_dim": VECTOR_DB_CONFIG.get("embedding_dim", 384)
            }
        )
        
        logger.info("Etymology cache initialized")
    
    async def store(self, word: str, result: EtymologyResult) -> bool:
        """
        Store etymology result in cache.
        
        Args:
            word: Word to cache
            result: Etymology result to store
            
        Returns:
            bool indicating success
        """
        try:
            # Convert result to dictionary
            data = {
                "word": word,
                "earliest_ancestors": result.earliest_ancestors,
                "transitions": [t.__dict__ for t in result.transitions],
                "sources": [s.__dict__ for s in result.sources],
                "confidence_score": result.confidence_score,
                "relations": {
                    word: [rel.__dict__ for rel in rels]
                    for word, rels in result.relations.items()
                }
            }
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[word],
                documents=[str(data)],  # Convert to string for storage
                metadatas=[{
                    "word": word,
                    "confidence": result.confidence_score
                }]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache etymology for {word}: {str(e)}")
            return False
    
    async def get(self, word: str) -> Optional[EtymologyResult]:
        """
        Retrieve etymology result from cache.
        
        Args:
            word: Word to retrieve
            
        Returns:
            EtymologyResult if found, None if not in cache
        """
        try:
            # Query ChromaDB
            results = self.collection.get(
                ids=[word],
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                return None
            
            # Parse stored data
            data = eval(results["documents"][0])  # Safe since we stored our own data
            
            # Reconstruct relations
            relations = {
                word: [WordRelation(**rel) for rel in rels]
                for word, rels in data.get("relations", {}).items()
            }
            
            # Reconstruct EtymologyResult
            return EtymologyResult(
                word=data["word"],
                earliest_ancestors=data["earliest_ancestors"],
                transitions=[
                    LanguageTransition(**t)
                    for t in data["transitions"]
                ],
                sources=[
                    EtymologySource(**s)
                    for s in data["sources"]
                ],
                confidence_score=data["confidence_score"],
                relations=relations
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve etymology for {word}: {str(e)}")
            return None
    
    def clear(self) -> None:
        """Clear all cached data."""
        try:
            self.collection.delete(where={})
            logger.info("Etymology cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear etymology cache: {str(e)}") 
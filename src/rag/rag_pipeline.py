"""
Core implementation of the RAG pipeline for etymology data.

This module provides a flexible, extensible system for storing and retrieving
etymological data using various backend storage systems. The design emphasizes
easy swapping of storage backends and future extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import numpy as np
from numpy.typing import NDArray
import os

import chromadb
from loguru import logger

from src.agents.research_agent import EtymologyResult
from src.models.stem_models import Stem, StemAnalysis
from src.agents.similarity_agent import SimilarityResult
from src.config import VECTOR_DB_CONFIG, VECTOR_DB_DIR


@dataclass
class EtymologyDocument:
    """
    Document structure for storing etymology data.
    
    This class represents the core data structure for etymology information
    in the RAG system. It's designed to be serializable and storage-agnostic.
    
    Attributes:
        word: The target word
        etymology_data: Structured etymology information
        embeddings: Word and context embeddings
        metadata: Additional information and provenance
        last_updated: Timestamp of last update
    """
    word: str
    etymology_data: Dict
    embeddings: Optional[Dict[str, NDArray]] = None
    metadata: Dict = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = {
            "word": self.word,
            "etymology_data": {
                'word': self.etymology_data['word'],
                'earliest_ancestors': self.etymology_data['earliest_ancestors'],
                'transitions': [
                    {
                        'from_lang': t['from_lang'],
                        'to_lang': t['to_lang'],
                        'from_word': t['from_word'],
                        'to_word': t['to_word'],
                        'approximate_date': t.get('approximate_date'),
                        'notes': t.get('notes'),
                        'semantic_changes': t.get('semantic_changes', []),
                        'parallel_transitions': []  # Avoid circular references
                    }
                    for t in self.etymology_data['transitions']
                ],
                'sources': [
                    {
                        'name': s['name'],
                        'url': s['url'],
                        'accessed_date': s['accessed_date'],
                        'confidence': s.get('confidence', 1.0)
                    }
                    for s in self.etymology_data['sources']
                ],
                'confidence_score': self.etymology_data['confidence_score'],
                'relations': self.etymology_data.get('relations', {})
            },
            "metadata": self.metadata,
            "last_updated": self.last_updated
        }
        if self.embeddings:
            # Convert numpy arrays to lists for JSON serialization
            data["embeddings"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.embeddings.items()
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EtymologyDocument':
        """Create from dictionary representation."""
        # Convert embeddings back to numpy arrays if present
        embeddings = None
        if "embeddings" in data:
            embeddings = {
                k: (np.array(v) if isinstance(v, list) else v)
                for k, v in data["embeddings"].items()
            }
        
        # Create new instance
        return cls(
            word=data["word"],
            etymology_data=data["etymology_data"],
            embeddings=embeddings,
            metadata=data.get("metadata", {}),
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )


@dataclass
class RetrievalResult:
    """
    Result of a RAG pipeline retrieval operation.
    
    Attributes:
        document: Retrieved etymology document if found
        similarity_score: Similarity score if retrieved by similarity
        source: Source of the retrieved data
    """
    document: Optional[EtymologyDocument]
    similarity_score: Optional[float] = None
    source: str = "direct_lookup"


class StorageBackend(ABC):
    """
    Abstract base class for RAG storage backends.
    
    This class defines the interface that all storage backends must implement.
    New storage systems can be added by implementing this interface.
    """
    
    @abstractmethod
    async def store(
        self,
        document: EtymologyDocument,
        overwrite: bool = False
    ) -> bool:
        """Store an etymology document."""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        word: str,
        threshold: float = 0.0
    ) -> Optional[RetrievalResult]:
        """Retrieve etymology data for a word."""
        pass
    
    @abstractmethod
    async def retrieve_similar(
        self,
        embedding: NDArray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[RetrievalResult]:
        """Retrieve similar documents by embedding."""
        pass
    
    @abstractmethod
    async def exists(self, word: str, collection: str = "etymology") -> bool:
        """
        Check if word exists in ChromaDB.
        
        Args:
            word: Word to check
            collection: Which collection to check ("etymology" or "stems")
            
        Returns:
            bool indicating if word exists
        """
        pass


class ChromaBackend(StorageBackend):
    """
    ChromaDB implementation of the storage backend.
    
    Uses ChromaDB for both document storage and similarity search.
    Provides efficient vector storage and retrieval capabilities.
    """
    
    def __init__(self, persist_dir: Path = VECTOR_DB_DIR):
        """Initialize ChromaDB backend."""
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Get collection names from config
        etymology_collection = VECTOR_DB_CONFIG.get("etymology_collection", "etymologies")
        stems_collection = VECTOR_DB_CONFIG.get("stems_collection", "stems")
        
        # Create collections with proper settings
        self.etymology_collection = self.client.get_or_create_collection(
            name=etymology_collection,
            metadata={
                "hnsw:space": VECTOR_DB_CONFIG.get("similarity_metric", "cosine"),
                "embedding_dim": VECTOR_DB_CONFIG.get("embedding_dim", 384)
            }
        )
        
        self.stems_collection = self.client.get_or_create_collection(
            name=stems_collection,
            metadata={
                "hnsw:space": VECTOR_DB_CONFIG.get("similarity_metric", "cosine"),
                "embedding_dim": VECTOR_DB_CONFIG.get("embedding_dim", 384)
            }
        )
        
        # For backward compatibility
        self.collection = self.etymology_collection
        
        logger.info(f"Initialized ChromaDB backend at {persist_dir}")
    
    async def store(
        self,
        document: EtymologyDocument,
        overwrite: bool = False
    ) -> bool:
        """
        Store an etymology document in ChromaDB.
        
        Args:
            document: Document to store
            overwrite: Whether to overwrite existing data
            
        Returns:
            bool indicating success
        """
        try:
            # Check if document already exists
            if await self.exists(document.word) and not overwrite:
                return False
            
            # Convert document to dictionary and serialize
            doc_dict = document.to_dict()
            doc_str = json.dumps(doc_dict)
            
            # Extract word embedding if available
            embedding = None
            if document.embeddings and "word_embedding" in document.embeddings:
                embedding = document.embeddings["word_embedding"]
                if isinstance(embedding, dict) and "similar_words" in embedding:
                    # Create default embedding if only similarity data present
                    embedding = np.zeros(VECTOR_DB_CONFIG.get("embedding_dim", 384))
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
            
            # Store in ChromaDB
            try:
                self.etymology_collection.upsert(
                    ids=[document.word],
                    documents=[doc_str],
                    embeddings=[embedding] if embedding is not None else None,
                    metadatas=[{
                        "word": document.word,
                        "last_updated": document.last_updated,
                        "has_embedding": embedding is not None
                    }]
                )
                return True
            except Exception as e:
                logger.error(f"ChromaDB storage error for {document.word}: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error storing document for {document.word}: {str(e)}")
            return False
    
    async def retrieve(
        self,
        word: str,
        threshold: float = 0.0
    ) -> Optional[RetrievalResult]:
        """
        Retrieve etymology data for a word.
        
        Args:
            word: Word to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            RetrievalResult if found, None otherwise
        """
        try:
            # Query ChromaDB
            results = self.etymology_collection.get(
                ids=[word],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                logger.debug(f"No document found for {word}")
                return None
                
            try:
                # Parse the document string
                doc_str = results["documents"][0]
                if isinstance(doc_str, str):
                    # Remove any extra quotes
                    doc_str = doc_str.strip().strip("'\"")
                    doc_dict = json.loads(doc_str)
                else:
                    doc_dict = doc_str
                
                # Create EtymologyDocument
                document = EtymologyDocument.from_dict(doc_dict)
                
                # Add embedding if available
                if results.get("embeddings") and results["embeddings"][0] is not None:
                    document.embeddings = {
                        "word_embedding": np.array(results["embeddings"][0])
                    }
                
                # Get similarity score from metadata if available
                similarity_score = None
                if results.get("metadatas") and results["metadatas"][0]:
                    similarity_score = results["metadatas"][0].get("similarity_score")
                
                return RetrievalResult(
                    document=document,
                    similarity_score=similarity_score,
                    source="direct_lookup"
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse document for {word}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document for {word}: {str(e)}")
            return None
    
    async def retrieve_similar(
        self,
        embedding: NDArray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Retrieve similar documents by embedding.
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            results = self.etymology_collection.query(
                query_embeddings=embedding.reshape(-1).tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieval_results = []
            for doc, metadata, distance in zip(
                results["documents"],
                results["metadatas"],
                results["distances"]
            ):
                if distance > threshold:
                    continue
                    
                doc_dict = json.loads(doc)
                document = EtymologyDocument.from_dict(doc_dict)
                
                retrieval_results.append(
                    RetrievalResult(
                        document=document,
                        similarity_score=1 - distance,  # Convert distance to similarity
                        source="chroma_similarity"
                    )
                )
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def exists(self, word: str, collection: str = "etymology") -> bool:
        """
        Check if word exists in ChromaDB.
        
        Args:
            word: Word to check
            collection: Which collection to check ("etymology" or "stems")
            
        Returns:
            bool indicating if word exists
        """
        try:
            target_collection = (
                self.stems_collection if collection == "stems"
                else self.etymology_collection
            )
            results = target_collection.get(
                ids=[word],
                include=["metadatas"]
            )
            return bool(results["metadatas"])
        except Exception:
            return False

    async def store_stem_analysis(
        self,
        analysis: StemAnalysis,
        overwrite: bool = False
    ) -> bool:
        """
        Store stem analysis results.
        
        Args:
            analysis: Stem analysis to store
            overwrite: Whether to overwrite existing data
            
        Returns:
            bool indicating success
        """
        try:
            # Check if analysis already exists
            if await self.exists(analysis.word, collection="stems") and not overwrite:
                return False
            
            # Prepare document for storage
            doc_dict = analysis.to_dict()
            
            # Store each stem's embedding
            embeddings = []
            for stem in analysis.stems:
                if stem.embedding is not None:
                    embeddings.append(stem.embedding)
                else:
                    embeddings.append(np.zeros(VECTOR_DB_CONFIG["embedding_dim"]))
            
            # Average stem embeddings for document embedding
            if embeddings:
                doc_embedding = np.mean(embeddings, axis=0)
            else:
                doc_embedding = np.zeros(VECTOR_DB_CONFIG["embedding_dim"])
            
            # Ensure embedding is a list
            if isinstance(doc_embedding, np.ndarray):
                doc_embedding = doc_embedding.tolist()
            
            # Store in ChromaDB
            try:
                self.stems_collection.upsert(
                    ids=[analysis.word],
                    embeddings=[doc_embedding],
                    documents=[json.dumps(doc_dict)],
                    metadatas=[{
                        "word": analysis.word,
                        "confidence": analysis.confidence_score,
                        "stem_count": len(analysis.stems)
                    }]
                )
                return True
            except Exception as e:
                logger.error(f"ChromaDB storage error for stem analysis of {analysis.word}: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error storing stem analysis for {analysis.word}: {str(e)}")
            return False
    
    async def get_stem_analysis(
        self,
        word: str,
        threshold: float = 0.0
    ) -> Optional[StemAnalysis]:
        """
        Retrieve stem analysis for a word.
        
        Args:
            word: Word to retrieve analysis for
            threshold: Minimum similarity threshold
            
        Returns:
            StemAnalysis if found, None otherwise
        """
        try:
            logger.info(f"Retrieving stem analysis for word: {word}")
            results = self.stems_collection.get(
                ids=[word],
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                logger.warning(f"No stem analysis document found for word: {word}")
                return None
            
            try:
                # Get the document string/dict
                doc_data = results["documents"][0]
                logger.debug(f"Raw document data: {doc_data}")
                
                # Convert to dictionary if needed
                if isinstance(doc_data, str):
                    try:
                        # Clean the string if needed
                        if doc_data.startswith("'") and doc_data.endswith("'"):
                            doc_data = doc_data[1:-1]
                        if doc_data.startswith('"') and doc_data.endswith('"'):
                            doc_data = doc_data[1:-1]
                        logger.debug(f"Cleaned document string: {doc_data}")
                        
                        # Parse JSON string
                        doc_dict = json.loads(doc_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error at position {e.pos}: {e.msg}")
                        logger.error(f"Problem string: {doc_data[max(0, e.pos-50):min(len(doc_data), e.pos+50)]}")
                        return None
                elif isinstance(doc_data, dict):
                    doc_dict = doc_data
                else:
                    logger.error(f"Unexpected document type: {type(doc_data)}")
                    return None
                
                logger.debug(f"Working with document dictionary: {json.dumps(doc_dict, indent=2)}")
                
                # Process stems with enhanced validation
                stems = []
                if "stems" not in doc_dict:
                    logger.warning(f"No 'stems' field found in document for {word}")
                    # Try to construct from single stem if available
                    if any(key in doc_dict for key in ["stem", "text"]):
                        logger.info("Converting single stem to stems array")
                        stem_text = doc_dict.get("stem") or doc_dict.get("text", word)
                        doc_dict = {
                            "stems": [{
                                "text": stem_text,
                                "stem_type": doc_dict.get("stem_type", doc_dict.get("type", "root")),
                                "position": [0, len(stem_text)],
                                "language": {
                                    "origin": "Unknown",
                                    "family": "Unknown",
                                    "period": "Unknown"
                                },
                                "meaning": {
                                    "core": "Unknown",
                                    "extended": []
                                },
                                "etymology": {
                                    "development": "Unknown",
                                    "cognates": []
                                },
                                "morphology": {
                                    "allomorphs": [],
                                    "combinations": [],
                                    "restrictions": []
                                },
                                "examples": {
                                    "modern": [],
                                    "historical": [],
                                    "related_terms": []
                                },
                                "confidence": 0.5
                            }],
                            "confidence_score": doc_dict.get("confidence", 0.5),
                            "notes": "",
                            "metadata": {}
                        }
                
                for stem_data in doc_dict.get("stems", []):
                    try:
                        logger.debug(f"Processing stem data: {json.dumps(stem_data, indent=2)}")
                        
                        # Ensure stem_data is a dictionary
                        if not isinstance(stem_data, dict):
                            logger.warning(f"Invalid stem data type: {type(stem_data)}")
                            continue
                            
                        # Ensure all required fields exist with proper structure
                        processed_stem = {
                            "text": str(stem_data.get("text", "")),
                            "stem_type": str(stem_data.get("stem_type", stem_data.get("type", "root"))),
                            "position": tuple(stem_data.get("position", [0, len(str(stem_data.get("text", "")))])),
                            "language": {
                                "origin": str((stem_data.get("language", {}) or {}).get("origin", "Unknown")),
                                "family": str((stem_data.get("language", {}) or {}).get("family", "Unknown")),
                                "period": str((stem_data.get("language", {}) or {}).get("period", "Unknown"))
                            },
                            "meaning": {
                                "core": str((stem_data.get("meaning", {}) or {}).get("core", "Unknown")),
                                "extended": list((stem_data.get("meaning", {}) or {}).get("extended", []))
                            },
                            "etymology": {
                                "development": str((stem_data.get("etymology", {}) or {}).get("development", "Unknown")),
                                "cognates": list((stem_data.get("etymology", {}) or {}).get("cognates", []))
                            },
                            "morphology": {
                                "allomorphs": list((stem_data.get("morphology", {}) or {}).get("allomorphs", [])),
                                "combinations": list((stem_data.get("morphology", {}) or {}).get("combinations", [])),
                                "restrictions": list((stem_data.get("morphology", {}) or {}).get("restrictions", []))
                            },
                            "examples": {
                                "modern": list((stem_data.get("examples", {}) or {}).get("modern", [])),
                                "historical": list((stem_data.get("examples", {}) or {}).get("historical", [])),
                                "related_terms": list((stem_data.get("examples", {}) or {}).get("related_terms", []))
                            },
                            "confidence": float(stem_data.get("confidence", 0.5))
                        }
                        
                        logger.debug(f"Processed stem data: {json.dumps(processed_stem, indent=2)}")
                        stems.append(Stem(**processed_stem))
                        
                    except Exception as e:
                        logger.error(f"Error processing stem data: {str(e)}")
                        continue
                
                if not stems:
                    logger.warning(f"No valid stems found for {word}")
                    return None
                
                try:
                    analysis = StemAnalysis(
                        word=word,
                        stems=stems,
                        confidence_score=float(doc_dict.get("confidence_score", 0.5)),
                        notes=str(doc_dict.get("notes", "")),
                        metadata=dict(doc_dict.get("metadata", {}))
                    )
                    logger.info(f"Successfully created StemAnalysis for {word} with {len(stems)} stems")
                    return analysis
                except Exception as e:
                    logger.error(f"Error creating StemAnalysis: {str(e)}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error processing stem analysis for {word}: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving stem analysis for {word}: {str(e)}")
            return None
    
    async def find_similar_stems(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar stems by embedding.
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (stem, score) tuples
        """
        try:
            # Ensure embedding is a list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            results = self.stems_collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            similar_stems = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                similarity = 1 - distance  # Convert distance to similarity
                if similarity >= threshold:
                    doc_dict = json.loads(doc)
                    similar_stems.append((doc_dict["word"], float(similarity)))
            
            return similar_stems
            
        except Exception as e:
            logger.error(f"Error in stem similarity search: {str(e)}")
            return []


class RAGPipeline:
    """
    Main RAG pipeline for etymology data management.
    
    This class provides the high-level interface for storing and retrieving
    etymology data. It abstracts away the storage backend details and provides
    a clean API for the rest of the application.
    """
    
    def __init__(self, similarity_agent=None, cache_dir: str = None):
        """Initialize RAG pipeline."""
        # Initialize cache directory
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "../../data/vector_store")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize ChromaDB backend
        self.backend = ChromaBackend(Path(self.cache_dir))
        logger.info(f"Initialized ChromaDB backend at {self.cache_dir}")
        
        # Set up similarity agent
        self.similarity_agent = similarity_agent
        if self.similarity_agent:
            self.similarity_agent.set_rag_pipeline(self)
            logger.info("Initialized similarity agent with RAG pipeline")
    
    async def store_etymology(
        self,
        etymology_result: EtymologyResult,
        embeddings: Optional[Dict[str, Union[NDArray, 'SimilarityResult']]] = None,
        overwrite: bool = False
    ) -> bool:
        """
        Store etymology data in the RAG system.
        
        Args:
            etymology_result: Research results to store
            embeddings: Optional word and context embeddings or similarity results
            overwrite: Whether to overwrite existing data
            
        Returns:
            bool indicating success
        """
        # Convert embeddings to proper format
        processed_embeddings = {}
        if embeddings:
            for key, value in embeddings.items():
                if isinstance(value, np.ndarray):
                    processed_embeddings[key] = value
                elif hasattr(value, 'similar_words'):
                    # Handle SimilarityResult objects
                    processed_embeddings[key] = {
                        'similar_words': [
                            {
                                'word': word,
                                'language': lang,
                                'score': float(score)
                            }
                            for word, lang, score in value.similar_words
                        ],
                        'model_name': value.model_name,
                        'metric': value.metric,
                        'metadata': value.metadata
                    }
                else:
                    logger.warning(f"Skipping unknown embedding type for {key}: {type(value)}")
        
        document = EtymologyDocument(
            word=etymology_result.word,
            etymology_data={
                'word': etymology_result.word,
                'earliest_ancestors': etymology_result.earliest_ancestors,
                'transitions': [t.__dict__ for t in etymology_result.transitions],
                'sources': [s.__dict__ for s in etymology_result.sources],
                'confidence_score': etymology_result.confidence_score
            },
            embeddings=processed_embeddings,
            metadata={
                "confidence": etymology_result.confidence_score,
                "sources": [s.__dict__ for s in etymology_result.sources]
            }
        )
        
        return await self.backend.store(document, overwrite)
    
    async def get_etymology(
        self,
        word: str,
        threshold: float = 0.0
    ) -> Optional[RetrievalResult]:
        """
        Retrieve etymology data for a word.
        
        Args:
            word: Word to retrieve data for
            threshold: Minimum similarity threshold
            
        Returns:
            RetrievalResult if found, None otherwise
        """
        return await self.backend.retrieve(word, threshold)
    
    async def find_similar(
        self,
        embedding: NDArray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Find similar etymologies by embedding.
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult objects
        """
        return await self.backend.retrieve_similar(
            embedding,
            top_k,
            threshold
        )
    
    async def get_stem_analysis(
        self,
        word: str,
        threshold: float = 0.0
    ) -> Optional[StemAnalysis]:
        """
        Retrieve stem analysis for a word.
        
        Args:
            word: Word to retrieve analysis for
            threshold: Minimum similarity threshold
            
        Returns:
            StemAnalysis if found, None otherwise
        """
        return await self.backend.get_stem_analysis(word, threshold)
    
    async def store_stem_analysis(self, analysis: StemAnalysis) -> None:
        """
        Store stem analysis results.
        
        Args:
            analysis: StemAnalysis to store
        """
        await self.backend.store_stem_analysis(analysis)
    
    async def find_similar_stems(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar stems by embedding.
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (stem, score) tuples
        """
        return await self.backend.find_similar_stems(embedding, top_k, threshold) 
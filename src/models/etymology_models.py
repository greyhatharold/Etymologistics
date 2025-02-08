"""
Data models for etymology analysis.

This module contains the core data structures used throughout the etymology
analysis pipeline. These models are shared between different components
to ensure consistent data representation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
from datetime import datetime

class RelationType(Enum):
    """Types of relationships between words."""
    COGNATE = "cognate"  # Words with common ancestry
    BORROWING = "borrowing"  # Word borrowed from another language
    SEMANTIC = "semantic"  # Words with related meanings
    PHONETIC = "phonetic"  # Words with similar sounds
    COMPOUND = "compound"  # Word formed from multiple parts

@dataclass
class EtymologySource:
    """
    Source information for etymology data.
    
    Attributes:
        name: Name of the source
        url: URL where the information was found
        accessed_date: When the source was accessed
        confidence: Confidence score for this source
    """
    name: str
    url: str
    accessed_date: str
    confidence: float = 1.0

@dataclass
class WordRelation:
    """
    Represents a relationship between two words.
    
    Attributes:
        word: The related word
        language: Language of the related word
        relation_type: Type of relationship
        confidence: Confidence in the relationship (0-1)
        notes: Additional context about the relationship
        bidirectional: Whether the relationship applies both ways
    """
    word: str
    language: str
    relation_type: RelationType
    confidence: float
    notes: Optional[str] = None
    bidirectional: bool = True

@dataclass
class LanguageTransition:
    """
    Represents a transition between languages in etymology.
    
    Attributes:
        from_lang: Source language
        to_lang: Target language
        from_word: Word form in source language
        to_word: Word form in target language
        approximate_date: Approximate date of transition
        notes: Additional notes about the transition
        semantic_changes: List of meaning changes that occurred during this transition
        parallel_transitions: List of parallel language transitions that occurred around the same time
    """
    from_lang: str
    to_lang: str
    from_word: str
    to_word: str
    approximate_date: Optional[str] = None
    notes: Optional[str] = None
    semantic_changes: List[str] = field(default_factory=list)
    parallel_transitions: List['LanguageTransition'] = field(default_factory=list)

@dataclass
class EtymologyResult:
    """
    Complete etymology research result.
    
    Attributes:
        word: The researched word
        earliest_ancestors: List of (language, word) tuples for earliest known forms
        transitions: List of language transitions
        sources: List of sources consulted
        confidence_score: Overall confidence in the results
        relations: Dictionary mapping words to their relationships
    """
    word: str
    earliest_ancestors: List[Tuple[str, str]]
    transitions: List[LanguageTransition]
    sources: List[EtymologySource]
    confidence_score: float = 1.0
    relations: Dict[str, List[WordRelation]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "word": self.word,
            "earliest_ancestors": self.earliest_ancestors,
            "transitions": [t.__dict__ for t in self.transitions],
            "sources": [s.__dict__ for s in self.sources],
            "confidence_score": self.confidence_score,
            "relations": {
                word: [rel.__dict__ for rel in rels]
                for word, rels in self.relations.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EtymologyResult':
        """Create from dictionary representation."""
        relations = {
            word: [WordRelation(**rel) for rel in rels]
            for word, rels in data.get("relations", {}).items()
        }
        return cls(
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

@dataclass
class EtymologyResult:
    """
    Structured result of etymological research.
    
    Attributes:
        word: Target word that was researched
        earliest_ancestors: List of earliest known forms
        transitions: Chronological list of language transitions
        sources: Sources consulted for this information
        confidence_score: Overall confidence in the results (0-1)
        relations: Dictionary of related words and their relationships
        semantic_field: Set of words in the same semantic domain
        parallel_evolutions: List of words that evolved similarly
        compound_elements: Component parts if word is compound
    """
    word: str
    earliest_ancestors: List[Tuple[str, str]]  # (language, word_form)
    transitions: List[LanguageTransition]
    sources: List[EtymologySource]
    confidence_score: float
    relations: Dict[str, List[WordRelation]] = field(default_factory=dict)
    semantic_field: Set[str] = field(default_factory=set)
    parallel_evolutions: List[str] = field(default_factory=list)
    compound_elements: List[Tuple[str, str]] = field(default_factory=list)  # (element, meaning) 
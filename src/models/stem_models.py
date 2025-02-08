"""
Data models for morphological stem analysis.

This module contains the data structures used for representing and analyzing
word stems, roots, affixes, and their relationships.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class Stem:
    """
    Represents a morphological stem/root with its properties.
    
    Attributes:
        text: The actual text of the stem
        stem_type: Type of stem (prefix, suffix, root)
        position: Start and end indices in the word
        language: Origin language information
        meaning: Definition and semantic information
        etymology: Etymology information
        morphology: Morphological features
        examples: Example usages
        confidence: Confidence score for this analysis
    """
    text: str
    stem_type: str
    position: Tuple[int, int]
    language: Dict[str, str]
    meaning: Dict[str, str]
    etymology: Dict[str, str]
    morphology: Dict[str, List[str]]
    examples: Dict[str, List[str]]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "stem_type": self.stem_type,
            "position": list(self.position),
            "language": self.language,
            "meaning": self.meaning,
            "etymology": self.etymology,
            "morphology": self.morphology,
            "examples": self.examples,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stem':
        """Create from dictionary."""
        stem_type = data.get('stem_type', data.get('type'))
        if not stem_type:
            raise ValueError("Missing required field 'stem_type' or 'type'")
            
        return cls(
            text=data["text"],
            stem_type=stem_type,
            position=tuple(data["position"]),
            language=data["language"],
            meaning=data["meaning"],
            etymology=data["etymology"],
            morphology=data["morphology"],
            examples=data["examples"],
            confidence=data["confidence"]
        )


@dataclass
class StemAnalysis:
    """
    Container for stem analysis results.
    
    Attributes:
        word: The analyzed word
        stems: List of identified stems
        confidence_score: Overall confidence in the analysis
        notes: Additional analysis notes
        metadata: Additional metadata about the analysis
    """
    word: str
    stems: List[Stem]
    confidence_score: float
    notes: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    def get_stem_by_type(self, stem_type: str) -> List[Stem]:
        """Get all stems of a particular type."""
        return [s for s in self.stems if s.stem_type == stem_type]
    
    def get_stem_by_position(self, position: int) -> Optional[Stem]:
        """Get stem at a particular position in the word."""
        for stem in self.stems:
            if stem.position and stem.position[0] <= position <= stem.position[1]:
                return stem
        return None
    
    def get_related_stems(self, stem: Stem, threshold: float = 0.7) -> List[Stem]:
        """Get stems related to the given stem based on embedding similarity."""
        if not stem.embedding:
            return []
        
        related = []
        for other in self.stems:
            if other != stem and other.embedding is not None:
                similarity = np.dot(stem.embedding, other.embedding)
                if similarity >= threshold:
                    related.append(other)
        return related
    
    def get_semantic_fields(self) -> Dict[str, List[Stem]]:
        """Group stems by semantic field."""
        fields = {}
        for stem in self.stems:
            for field in stem.metadata.get("semantic_fields", []):
                if field not in fields:
                    fields[field] = []
                fields[field].append(stem)
        return fields
    
    def get_language_families(self) -> Dict[str, List[Stem]]:
        """Group stems by language family."""
        families = {}
        for stem in self.stems:
            family = stem.metadata.get("language_family")
            if family:
                if family not in families:
                    families[family] = []
                families[family].append(stem)
        return families
    
    def get_historical_periods(self) -> Dict[str, List[Stem]]:
        """Group stems by historical period."""
        periods = {}
        for stem in self.stems:
            period = stem.metadata.get("historical_period")
            if period:
                if period not in periods:
                    periods[period] = []
                periods[period].append(stem)
        return periods
    
    def get_morphological_patterns(self) -> Dict[str, List[str]]:
        """Extract common morphological patterns."""
        patterns = {
            "allomorphs": [],
            "combinations": [],
            "restrictions": []
        }
        for stem in self.stems:
            patterns["allomorphs"].extend(stem.metadata.get("allomorphs", []))
            patterns["combinations"].extend(stem.metadata.get("combinations", []))
            patterns["restrictions"].extend(stem.metadata.get("restrictions", []))
        return {k: list(set(v)) for k, v in patterns.items()}

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "word": self.word,
            "stems": [
                {
                    "text": stem.text,
                    "stem_type": stem.stem_type,
                    "language": stem.language,
                    "meaning": stem.meaning,
                    "position": stem.position,
                    "etymology": stem.etymology,
                    "similar_roots": stem.similar_roots,
                    "example_words": stem.examples,
                    "confidence": stem.confidence,
                    "metadata": stem.metadata
                }
                for stem in self.stems
            ],
            "confidence_score": self.confidence_score,
            "notes": self.notes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StemAnalysis':
        """Create from dictionary representation."""
        stems = [
            Stem(
                text=stem_data["text"],
                stem_type=stem_data["stem_type"],
                position=tuple(stem_data["position"]),
                language=stem_data["language"],
                meaning=stem_data["meaning"],
                etymology=stem_data["etymology"],
                morphology=stem_data["morphology"],
                examples=stem_data["examples"],
                confidence=stem_data["confidence"],
                embedding=np.array(stem_data.get("embedding", None)),
                similar_roots=stem_data.get("similar_roots", []),
                metadata=stem_data.get("metadata", {})
            )
            for stem_data in data["stems"]
        ]
        
        return cls(
            word=data["word"],
            stems=stems,
            confidence_score=data["confidence_score"],
            notes=data.get("notes"),
            metadata=data.get("metadata", {})
        ) 
"""
Tree Construction Agent for etymology visualization.

This module implements the TreeAgent class, responsible for converting
structured etymology data into a hierarchical tree structure
suitable for visualization of word evolution over time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from loguru import logger

from src.config import LANGUAGE_FAMILIES
from src.agents.research_agent import EtymologyResult, LanguageTransition
from src.models.etymology_models import RelationType, WordRelation


class NodeStyle(Enum):
    """Visual style types for different node categories."""
    ROOT = "root"  # Earliest known form
    INTERMEDIATE = "intermediate"  # Historical forms
    MODERN = "modern"  # Current/target word
    UNCERTAIN = "uncertain"  # Forms with low confidence
    RELATED = "related"  # Related words (cognates, etc.)
    PARALLEL = "parallel"  # Parallel evolution


class EdgeStyle(Enum):
    """Visual style types for different edge types."""
    EVOLUTION = "evolution"  # Direct evolution
    COGNATE = "cognate"  # Cognate relationship
    BORROWING = "borrowing"  # Borrowing relationship
    SEMANTIC = "semantic"  # Semantic relationship
    COMPOUND = "compound"  # Compound word relationship


@dataclass
class EtymologyNode:
    """
    Represents a node in the etymology tree.
    
    Attributes:
        word: The word form
        language: The language or language stage
        date: Approximate date or period
        style: Visual style category
        confidence: Confidence score for this etymology step (0-1)
        notes: Additional etymology information
        children: Child nodes (direct descendants)
        relations: Related nodes (cognates, etc.)
        parallel_nodes: Nodes showing parallel evolution
        semantic_changes: List of meaning changes
    """
    word: str
    language: str
    date: Optional[str] = None
    style: NodeStyle = NodeStyle.INTERMEDIATE
    confidence: float = 1.0
    notes: Optional[str] = None
    children: List['EtymologyNode'] = field(default_factory=list)
    relations: Dict[str, List['EtymologyNode']] = field(default_factory=dict)
    parallel_nodes: List['EtymologyNode'] = field(default_factory=list)
    semantic_changes: List[str] = field(default_factory=list)
    
    def add_child(self, child: 'EtymologyNode') -> None:
        """Add a direct descendant node."""
        self.children.append(child)
    
    def add_relation(self, relation_type: RelationType, node: 'EtymologyNode') -> None:
        """Add a related node with specified relationship type."""
        if relation_type.value not in self.relations:
            self.relations[relation_type.value] = []
        self.relations[relation_type.value].append(node)
    
    def add_parallel(self, node: 'EtymologyNode') -> None:
        """Add a node showing parallel evolution."""
        self.parallel_nodes.append(node)
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for visualization."""
        return {
            "id": f"{self.language}_{self.word}",
            "word": self.word,
            "language": self.language,
            "date": self.date,
            "style": self.style.value,
            "confidence": self.confidence,
            "notes": self.notes,
            "semantic_changes": self.semantic_changes,
            "children": [child.to_dict() for child in self.children],
            "relations": {
                rel_type: [node.to_dict() for node in nodes]
                for rel_type, nodes in self.relations.items()
            },
            "parallel_nodes": [node.to_dict() for node in self.parallel_nodes]
        }


class TreeAgent:
    """
    Agent responsible for constructing etymology trees.
    
    Creates a hierarchical tree structure showing word evolution over time,
    including relationships and parallel developments.
    """
    
    def __init__(self):
        """Initialize the tree agent with necessary configurations."""
        self.known_languages = {
            family.name: family.languages 
            for family in LANGUAGE_FAMILIES.values()
        }
        logger.info("TreeAgent initialized")
    
    def construct_tree(self, etymology_result: EtymologyResult) -> EtymologyNode:
        """
        Construct an etymology tree from research results.
        
        Creates a hierarchical tree showing word evolution through time,
        including related words and parallel developments.
        
        Args:
            etymology_result: Structured etymology data
            
        Returns:
            Root node of the constructed tree
        """
        # Handle empty results
        if not etymology_result:
            raise ValueError("Etymology data cannot be None")
            
        # Initialize with empty lists if None
        transitions = etymology_result.transitions or []
        earliest_ancestors = etymology_result.earliest_ancestors or []
        
        if not transitions and not earliest_ancestors:
            # Create a single node for the word if no etymology found
            return EtymologyNode(
                word=etymology_result.word,
                language="Unknown",
                style=NodeStyle.UNCERTAIN,
                confidence=0.0,
                notes="No etymology data found"
            )
        
        # Sort transitions chronologically
        sorted_transitions = self._sort_transitions(transitions)
        
        # Create root node from earliest ancestor
        if earliest_ancestors:
            lang, word = earliest_ancestors[0]
            root = EtymologyNode(
                word=word,
                language=lang,
                date=self._get_approximate_date(lang),
                style=NodeStyle.ROOT,
                confidence=etymology_result.confidence_score,
                notes="Earliest known form"
            )
        else:
            # Use first transition as root if no earliest ancestor
            first_trans = sorted_transitions[0]
            root = EtymologyNode(
                word=first_trans.from_word,
                language=first_trans.from_lang,
                date=first_trans.approximate_date,
                style=NodeStyle.ROOT,
                confidence=etymology_result.confidence_score,
                notes=first_trans.notes
            )
        
        # Build tree structure
        current_node = root
        processed = {(root.language, root.word)}
        
        # Process main evolution path
        for transition in sorted_transitions:
            # Skip if already processed to avoid cycles
            if (transition.from_lang, transition.from_word) in processed:
                continue
                
            # Create intermediate node if needed
            if current_node.word != transition.from_word:
                intermediate = EtymologyNode(
                    word=transition.from_word,
                    language=transition.from_lang,
                    date=transition.approximate_date,
                    style=NodeStyle.INTERMEDIATE,
                    confidence=etymology_result.confidence_score,
                    notes=transition.notes,
                    semantic_changes=transition.semantic_changes
                )
                current_node.add_child(intermediate)
                current_node = intermediate
                processed.add((intermediate.language, intermediate.word))
            
            # Create target node
            is_final = transition.to_word == etymology_result.word
            target = EtymologyNode(
                word=transition.to_word,
                language=transition.to_lang,
                date=transition.approximate_date,
                style=NodeStyle.MODERN if is_final else NodeStyle.INTERMEDIATE,
                confidence=etymology_result.confidence_score,
                notes=transition.notes,
                semantic_changes=transition.semantic_changes
            )
            current_node.add_child(target)
            current_node = target
            processed.add((target.language, target.word))
            
            # Process parallel transitions
            for parallel in transition.parallel_transitions:
                parallel_node = EtymologyNode(
                    word=parallel.to_word,
                    language=parallel.to_lang,
                    date=parallel.approximate_date,
                    style=NodeStyle.PARALLEL,
                    confidence=etymology_result.confidence_score,
                    notes=parallel.notes,
                    semantic_changes=parallel.semantic_changes
                )
                current_node.add_parallel(parallel_node)
        
        # Add relationships
        for rel_word, relations in etymology_result.relations.items():
            for relation in relations:
                related_node = EtymologyNode(
                    word=relation.word,
                    language=relation.language,
                    style=NodeStyle.RELATED,
                    confidence=relation.confidence,
                    notes=relation.notes
                )
                root.add_relation(relation.relation_type, related_node)
        
        return root
    
    def _sort_transitions(
        self,
        transitions: List[LanguageTransition]
    ) -> List[LanguageTransition]:
        """Sort transitions chronologically based on language stages."""
        
        def get_stage_score(lang: str) -> int:
            """Get numeric score for language stage for sorting."""
            lang = lang.lower()
            if "proto" in lang:
                return 0
            if "old" in lang:
                return 1
            if "middle" in lang:
                return 2
            if "early" in lang:
                return 3
            if "modern" in lang:
                return 4
            return 5
        
        return sorted(
            transitions,
            key=lambda t: (
                get_stage_score(t.from_lang),
                get_stage_score(t.to_lang)
            )
        )
    
    def _get_approximate_date(self, language: str) -> Optional[str]:
        """Get approximate date range for a language stage."""
        language = language.lower()
        if "proto" in language:
            return "Before 500 BCE"
        if "old" in language:
            return "500-1100 CE"
        if "middle" in language:
            return "1100-1500 CE"
        if "early" in language:
            return "1500-1800 CE"
        if "modern" in language:
            return "1800 CE-Present"
        return None 
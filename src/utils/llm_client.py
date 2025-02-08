"""
LLM client for morphological analysis.

This module provides a client for interacting with language models
to perform morphological analysis of words.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from openai import AsyncOpenAI
from loguru import logger

@dataclass
class StemData:
    """
    Container for stem analysis data from LLM.
    
    Attributes:
        text: The stem text
        stem_type: Type of stem (prefix, suffix, root)
        language: Origin language
        meaning: Definition/meaning
        position: Position in word (start, end)
        etymology: Etymology information
        examples: Example words using this stem
        confidence: Confidence score
    """
    text: str
    stem_type: str
    language: str
    meaning: str
    position: tuple[int, int]
    etymology: Optional[str] = None
    examples: List[str] = None
    confidence: float = 1.0


@dataclass
class MorphologyResponse:
    """
    Container for morphological analysis response.
    
    Attributes:
        stems: List of analyzed stems
        notes: Additional analysis notes
        timestamp: Analysis timestamp
        word_formation_pattern: Overall word formation pattern
        morphological_features: Notable morphological features
        development_insights: Historical development insights
        usage_patterns: Modern usage patterns and restrictions
    """
    stems: List[Dict[str, Any]]
    notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(datetime.UTC).isoformat())
    word_formation_pattern: Optional[str] = None
    morphological_features: List[str] = field(default_factory=list)
    development_insights: List[str] = field(default_factory=list)
    usage_patterns: List[str] = field(default_factory=list)


class LLMClient:
    """Client for LLM-based morphological analysis."""
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize LLM client."""
        self.model = model
        logger.info("LLM client initialized")
    
    def _create_stem_prompt(self, word: str, etymology: str) -> str:
        """
        Create prompt for stem analysis.
        
        Args:
            word: Word to analyze
            etymology: Etymology data to help analysis
            
        Returns:
            Analysis prompt
        """
        self.log_method_call("_create_stem_prompt", word=word, etymology_id=id(etymology))
        
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

Example response format:
{{
  "stems": [
    {{
      "text": "di-",
      "stem_type": "prefix",
      "position": [0, 2],
      "language": {{
        "origin": "Greek",
        "family": "Indo-European",
        "period": "Ancient"
      }},
      "meaning": {{
        "core": "two",
        "extended": ["double", "twice"]
      }},
      "etymology": {{
        "development": "From Greek di-",
        "cognates": ["dy-", "dis-"]
      }},
      "morphology": {{
        "allomorphs": ["dy-"],
        "combinations": ["di-sect"],
        "restrictions": []
      }},
      "examples": {{
        "modern": ["divide", "diverge"],
        "historical": ["dicho-"],
        "related_terms": []
      }},
      "confidence": 0.9
    }},
    {{
      "text": "-chot-",
      "stem_type": "root",
      "position": [2, 6],
      "language": {{
        "origin": "Greek",
        "family": "Indo-European",
        "period": "Ancient"
      }},
      "meaning": {{
        "core": "cut",
        "extended": ["divide", "split"]
      }},
      "etymology": {{
        "development": "From Greek khot-",
        "cognates": ["tom-"]
      }},
      "morphology": {{
        "allomorphs": ["tom-"],
        "combinations": ["anatomy"],
        "restrictions": []
      }},
      "examples": {{
        "modern": ["tome", "atom"],
        "historical": ["tomos"],
        "related_terms": []
      }},
      "confidence": 0.8
    }},
    {{
      "text": "-omy",
      "stem_type": "suffix",
      "position": [6, 9],
      "language": {{
        "origin": "Greek",
        "family": "Indo-European",
        "period": "Ancient"
      }},
      "meaning": {{
        "core": "process",
        "extended": ["study of", "cutting"]
      }},
      "etymology": {{
        "development": "From Greek -omia",
        "cognates": ["-tomy"]
      }},
      "morphology": {{
        "allomorphs": ["-tomy"],
        "combinations": ["anatomy"],
        "restrictions": []
      }},
      "examples": {{
        "modern": ["anatomy", "economy"],
        "historical": ["tomia"],
        "related_terms": []
      }},
      "confidence": 0.9
    }}
  ],
  "confidence_score": 0.85,
  "notes": "Clear Greek origin with well-documented morphological components",
  "metadata": {{
    "etymology_sources": ["Ancient Greek lexicons", "Historical linguistics"],
    "analysis_date": "{datetime.now().isoformat()}",
    "word_formation_pattern": "prefix + root + suffix",
    "morphological_features": [
      "compound formation",
      "derivational morphology",
      "Greek scientific terminology"
    ],
    "development_insights": [
      "Follows standard Greek scientific word formation",
      "Components retain original meanings"
    ],
    "usage_patterns": [
      "Common in scientific terminology",
      "Used metaphorically in general language"
    ]
  }}
}}

Consider:
1. Common prefixes/suffixes in the word's origin language
2. Morphological patterns in related words
3. Historical sound changes
4. Semantic development
5. Word formation rules

Provide detailed analysis with high confidence for clear components,
and lower confidence scores for inferred elements.

IMPORTANT: Ensure the response is a valid JSON object with all required fields and proper nesting."""

        logger.debug("Created stem analysis prompt", 
                    word=word,
                    transitions_count=len(transitions),
                    prompt_length=len(prompt))
        return prompt

    async def analyze_morphology(self, word: str, etymology: str) -> Dict[str, Any]:
        """
        Analyze morphological structure of a word.
        
        Args:
            word: Word to analyze
            etymology: Etymology data to help analysis
            
        Returns:
            Analysis result
        """
        try:
            # Create analysis prompt
            prompt = self._create_stem_prompt(word, etymology)
            
            # Get LLM response
            response = await self.llm.agenerate(prompt)
            response_text = response.text
            logger.debug(f"Raw LLM response: {response_text}")
            
            try:
                # Clean response text if needed
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1]
                if response_text.endswith("```"):
                    response_text = response_text.rsplit("```", 1)[0]
                response_text = response_text.strip()
                
                # Parse JSON response
                try:
                    result = json.loads(response_text)
                    logger.debug(f"Parsed LLM response: {json.dumps(result, indent=2)}")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                    logger.error(f"Invalid JSON near: {response_text[max(0, e.pos-50):min(len(response_text), e.pos+50)]}")
                    # Return a basic structure
                    return {
                        "stems": [{
                            "text": word,
                            "stem_type": "root",
                            "position": [0, len(word)],
                            "language": {"origin": "Unknown", "family": "Unknown", "period": "Unknown"},
                            "meaning": {"core": "Unknown", "extended": []},
                            "etymology": {"development": "Unknown", "cognates": []},
                            "morphology": {"allomorphs": [], "combinations": [], "restrictions": []},
                            "examples": {"modern": [], "historical": [], "related_terms": []},
                            "confidence": 0.5
                        }],
                        "confidence_score": 0.5,
                        "notes": "Failed to parse LLM response",
                        "metadata": {
                            "error": str(e),
                            "analysis_date": datetime.now().isoformat()
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in morphology analysis: {str(e)}")
            return None
    
    def _validate_stem_data(self, data: Dict) -> bool:
        """
        Validate stem analysis data.
        
        Args:
            data: Stem data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Handle legacy field names
            if "stem" in data and "text" not in data:
                data["text"] = data.pop("stem")
            if "type" in data and "stem_type" not in data:
                data["stem_type"] = data.pop("type")
            
            # Convert simplified structure to full structure if needed
            if "components" in data:
                # Create proper stem structure from simplified data
                components = data.pop("components")
                if isinstance(components, list) and len(components) > 0:
                    if "text" not in data:
                        data["text"] = components[0]  # Use first component as text
                    if "stem_type" not in data:
                        data["stem_type"] = "root"  # Default to root if not specified
                    data["position"] = [0, len(data["text"])]
                    data["language"] = {"origin": "", "period": "", "family": ""}
                    data["meaning"] = {"core": "", "extended": [], "semantic_fields": []}
                    data["etymology"] = {"development": "", "cognates": [], "semantic_changes": []}
                    data["morphology"] = {"allomorphs": [], "combinations": components, "restrictions": []}
                    data["examples"] = {"modern": [], "historical": [], "related_terms": []}
            
            # Check required fields and types with detailed logging
            required_fields = {
                "text": str,
                "stem_type": str,
                "position": list,
                "language": dict,
                "meaning": dict,
                "etymology": dict,
                "morphology": dict,
                "examples": dict,
                "confidence": (int, float)  # Allow both int and float
            }
            
            # Initialize missing fields with defaults
            for field, field_type in required_fields.items():
                if field not in data:
                    if field == "confidence":
                        data[field] = 0.8
                    elif field == "stem_type":
                        data[field] = "root"
                    elif field == "position":
                        data[field] = [0, len(data.get("text", ""))]
                    elif field_type == dict:
                        data[field] = {}
                    elif field_type == list:
                        data[field] = []
                    else:
                        data[field] = ""
            
            # Validate types and convert if possible
            for field, field_type in required_fields.items():
                if not isinstance(data[field], field_type):
                    if field == "confidence" and isinstance(data[field], str):
                        try:
                            if data[field].endswith("%"):
                                data[field] = float(data[field].rstrip("%")) / 100
                            elif data[field].lower() == "high":
                                data[field] = 0.9
                            elif data[field].lower() == "medium":
                                data[field] = 0.7
                            elif data[field].lower() == "low":
                                data[field] = 0.5
                            else:
                                data[field] = float(data[field])
                        except:
                            data[field] = 0.8
                    else:
                        logger.warning(f"Invalid type for {field}: expected {field_type}, got {type(data[field])}")
                        return False
            
            # Validate stem_type value
            if data["stem_type"] not in ["prefix", "root", "suffix"]:
                logger.warning(f"Invalid stem_type value: {data['stem_type']}")
                data["stem_type"] = "root"  # Default to root
            
            # Validate nested structures
            nested_fields = {
                "language": ["origin", "period", "family"],
                "meaning": ["core", "extended", "semantic_fields"],
                "etymology": ["development", "cognates", "semantic_changes"],
                "morphology": ["allomorphs", "combinations", "restrictions"],
                "examples": ["modern", "historical", "related_terms"]
            }
            
            for struct_name, required_keys in nested_fields.items():
                struct = data.get(struct_name, {})
                if not isinstance(struct, dict):
                    struct = {}
                
                # Initialize missing keys
                for key in required_keys:
                    if key not in struct:
                        if key in ["core", "development", "origin", "period", "family"]:
                            struct[key] = ""
                        else:
                            struct[key] = []
                    elif isinstance(struct[key], str) and key not in ["core", "development", "origin", "period", "family"]:
                        struct[key] = [struct[key]]  # Convert single string to list
                
                data[struct_name] = struct
            
            # Validate position format
            if not (isinstance(data["position"], list) and len(data["position"]) == 2 
                   and all(isinstance(x, (int, float)) for x in data["position"])):
                data["position"] = [0, len(data["text"])]
            
            # Validate confidence score
            if not 0 <= float(data["confidence"]) <= 1:
                data["confidence"] = 0.8
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
    
    async def _query_llm(
        self,
        prompt: str,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Query the LLM with proper formatting.
        
        Args:
            prompt: The analysis prompt
            response_format: Optional response format specification
            
        Returns:
            LLM response text
        """
        try:
            client = AsyncOpenAI()
            messages = [{
                "role": "system",
                "content": """You are a linguistic analysis assistant specializing in morphological analysis and etymology.
                Always provide responses in valid JSON format with properly quoted property names and string values.
                Ensure the response includes a 'stems' array with at least one stem analysis."""
            }, {
                "role": "user",
                "content": prompt
            }]
            
            # Add response format if specified
            kwargs = {
                "model": "gpt-4o",  # Use consistent model
                "messages": messages,
                "temperature": 0.2,  # Lower temperature for more consistent analysis
                "seed": 42  # Use consistent seed for reproducibility
            }
            if response_format:
                kwargs["response_format"] = response_format
            
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            # Validate JSON structure
            try:
                data = json.loads(content)
                if "stems" not in data or not data["stems"]:
                    # If missing stems, try to fix the structure
                    if isinstance(data, dict):
                        data = {"stems": [data]}
                        return json.dumps(data)
                return content
            except json.JSONDecodeError:
                # Try to extract JSON from response if not properly formatted
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json_match.group(0)
                raise ValueError("Could not extract valid JSON from response")
            
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            raise ValueError(f"LLM query failed: {str(e)}") 
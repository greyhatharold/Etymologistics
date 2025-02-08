"""
Data source abstractions for etymology research.

This module provides interfaces and implementations for different etymology data sources.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import aiohttp
import os
import json
from datetime import datetime
from pathlib import Path
import re
from bs4 import BeautifulSoup
from loguru import logger

from src.config import API_ENDPOINTS, SCRAPING_CONFIG
from src.models.etymology_models import EtymologyResult, EtymologySource, LanguageTransition
from src.utils.browser import Browser

class EtymologyDataSource(ABC):
    """Abstract base class for etymology data sources."""
    
    @abstractmethod
    async def query(self, word: str) -> Optional[EtymologyResult]:
        """Query this source for etymology data about a word."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Get the name of this data source."""
        pass

    @property
    @abstractmethod
    def confidence_score(self) -> float:
        """Get the confidence score for this data source."""
        pass

class LocalDatasetSource(EtymologyDataSource):
    """Local etymonline dataset source."""
    
    def __init__(self, dataset_path: Path):
        self.data = {}
        try:
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                    # Convert list to dictionary for faster lookups
                    self.data = {
                        entry["word"].lower(): entry
                        for entry in entries
                        if isinstance(entry, dict) and "word" in entry
                    }
                logger.info(f"Loaded {len(self.data)} entries from etymonline dataset")
        except Exception as e:
            logger.error(f"Failed to load etymonline dataset: {str(e)}")

    @property
    def source_name(self) -> str:
        return "Etymonline Dataset"

    @property
    def confidence_score(self) -> float:
        return 0.9

    async def query(self, word: str) -> Optional[EtymologyResult]:
        word = word.lower()
        if word not in self.data:
            return None
            
        entry = self.data[word]
        result = self._parse_entry(word, entry)
        return result  # Return the parsed result directly, no need to await
    
    def _parse_entry(self, word: str, entry: Dict) -> EtymologyResult:
        transitions = []
        ancestors = []
        
        etymology_text = entry.get("text", "")
        
        # Parse PIE roots
        pie_roots = re.findall(r"PIE root \*([^,\s]+)", etymology_text)
        if pie_roots:
            ancestors.extend([("Proto-Indo-European", f"*{root}") for root in pie_roots])
        
        # Parse transitions
        transition_patterns = [
            (r"from ([^,]+) \*?([^,\s]+)", "borrowed"),
            (r"cognate with ([^,]+) \*?([^,\s]+)", "cognate"),
            (r"from ([^,]+) base \*?([^,\s]+)", "derived")
        ]
        
        for pattern, trans_type in transition_patterns:
            matches = re.finditer(pattern, etymology_text)
            for match in matches:
                lang, word_form = match.groups()
                transitions.append(LanguageTransition(
                    from_lang=lang.strip(),
                    to_lang="English",
                    from_word=word_form.strip(),
                    to_word=word,
                    notes=f"{trans_type} relationship"
                ))
        
        # Add default transition if none found
        if not transitions and not ancestors:
            # Try to extract Old English or other language origins
            old_eng_match = re.search(r"Old English ([^,\s]+)", etymology_text)
            if old_eng_match:
                transitions.append(LanguageTransition(
                    from_lang="Old English",
                    to_lang="English",
                    from_word=old_eng_match.group(1).strip(),
                    to_word=word,
                    notes="direct descendant"
                ))
        
        return EtymologyResult(
            word=word,
            earliest_ancestors=ancestors,
            transitions=transitions,
            sources=[EtymologySource(
                name=self.source_name,
                url=f"https://www.etymonline.com/word/{word}",
                accessed_date=datetime.now().isoformat(),
                confidence=self.confidence_score
            )],
            confidence_score=self.confidence_score
        )

class WiktionaryAPISource(EtymologyDataSource):
    """Wiktionary API data source."""

    @property
    def source_name(self) -> str:
        return "Wiktionary"

    @property
    def confidence_score(self) -> float:
        return 0.8

    async def query(self, word: str) -> Optional[EtymologyResult]:
        params = {
            "action": "parse",
            "page": word,
            "format": "json",
            "prop": "wikitext|sections",
            "origin": "*"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(API_ENDPOINTS["wiktionary"].base_url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    if "error" in data:
                        return None
                    
                    wikitext = data["parse"]["wikitext"]["*"]
                    parsed_data = self._parse_wikitext(wikitext)
                    
                    return EtymologyResult(
                        word=word,
                        earliest_ancestors=parsed_data["ancestors"],
                        transitions=parsed_data["transitions"],
                        sources=[EtymologySource(
                            name=self.source_name,
                            url=f"https://en.wiktionary.org/wiki/{word}",
                            accessed_date=datetime.now().isoformat(),
                            confidence=self.confidence_score
                        )],
                        confidence_score=self.confidence_score
                    )
                    
        except Exception as e:
            logger.error(f"Error querying Wiktionary API: {str(e)}")
            return None

    def _parse_wikitext(self, wikitext: str) -> Dict:
        transitions = []
        ancestors = []
        
        # Extract etymology section
        etymology_text = ""
        in_etymology = False
        for line in wikitext.split('\n'):
            if line.startswith('==Etymology=='):
                in_etymology = True
            elif in_etymology and line.startswith('=='):
                break
            elif in_etymology:
                etymology_text += line + '\n'
        
        if not etymology_text:
            return {"ancestors": [], "transitions": []}
        
        # Parse language transitions
        lang_patterns = [
            (r"From {{etyl\|([^}]+)}}.*?{{m\|[^}]+\|([^}]+)}}", "from"),
            (r"borrowed from {{etyl\|([^}]+)}}.*?{{m\|[^}]+\|([^}]+)}}", "borrowed"),
            (r"from {{der\|en\|([^}]+)}}.*?{{m\|[^}]+\|([^}]+)}}", "derived")
        ]
        
        for pattern, transition_type in lang_patterns:
            matches = re.finditer(pattern, etymology_text)
            for match in matches:
                lang, word = match.groups()
                if transition_type == "from":
                    ancestors.append((lang.strip(), word.strip()))
                transitions.append(LanguageTransition(
                    from_lang=lang.strip(),
                    to_lang="English",
                    from_word=word.strip(),
                    to_word="",
                    notes=f"Transition type: {transition_type}"
                ))
        
        return {
            "ancestors": ancestors,
            "transitions": transitions
        }

class EtymonlineAPISource(EtymologyDataSource):
    """Etymonline API data source."""

    @property
    def source_name(self) -> str:
        return "Etymonline API"

    @property
    def confidence_score(self) -> float:
        return 0.85

    async def query(self, word: str) -> Optional[EtymologyResult]:
        api_key = os.getenv(API_ENDPOINTS["etymonline"].env_key_name)
        if not api_key or api_key.startswith("your_"):
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": SCRAPING_CONFIG["user_agent"]
                }
                url = f"{API_ENDPOINTS['etymonline'].base_url}/word/{word}"
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None
                        
                    data = await response.json()
                    parsed_data = self._parse_response(data)
                    
                    return EtymologyResult(
                        word=word,
                        earliest_ancestors=parsed_data["ancestors"],
                        transitions=parsed_data["transitions"],
                        sources=[EtymologySource(
                            name=self.source_name,
                            url=f"https://www.etymonline.com/word/{word}",
                            accessed_date=datetime.now().isoformat(),
                            confidence=self.confidence_score
                        )],
                        confidence_score=self.confidence_score
                    )
                    
        except Exception as e:
            logger.error(f"Failed to query Etymonline API: {str(e)}")
            return None

    def _parse_response(self, data: Dict) -> Dict:
        transitions = []
        ancestors = []
        
        for entry in data.get("results", []):
            text = entry.get("text", "")
            
            if "from PIE root" in text.lower():
                pie_match = re.search(r"from PIE root \*([^,]+)", text)
                if pie_match:
                    ancestors.append(("Proto-Indo-European", f"*{pie_match.group(1)}"))
            
            for match in re.finditer(r"from ([A-Z][a-zA-Z\s-]+) _?([^,]+)_?", text):
                lang, word = match.groups()
                transitions.append(LanguageTransition(
                    from_lang=lang.strip(),
                    to_lang="English",
                    from_word=word.strip(),
                    to_word=entry.get("word", "")
                ))
        
        return {
            "ancestors": ancestors,
            "transitions": transitions
        }

class WebScrapingSource(EtymologyDataSource):
    """Web scraping data source using browser automation."""

    def __init__(self):
        self.browser = None

    @property
    def source_name(self) -> str:
        return "Web Scraping"

    @property
    def confidence_score(self) -> float:
        return 0.6

    async def query(self, word: str) -> Optional[EtymologyResult]:
        try:
            # Use browser as context manager for better resource management
            async with Browser() as browser:
                page = await browser.create_page()
                await page.goto(f"https://www.etymonline.com/word/{word}")
                
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                etymology_div = soup.find('div', class_='word--C9UPa')
                if not etymology_div:
                    return None
                    
                etymology_text = etymology_div.get_text()
                
                transitions = []
                ancestors = []
                
                pie_roots = re.findall(r"PIE root \*([^,\s]+)", etymology_text)
                if pie_roots:
                    ancestors.extend([("Proto-Indo-European", f"*{root}") for root in pie_roots])
                
                transition_patterns = [
                    (r"from ([^,]+) \*?([^,\s]+)", "borrowed"),
                    (r"cognate with ([^,]+) \*?([^,\s]+)", "cognate"),
                    (r"from ([^,]+) base \*?([^,\s]+)", "derived")
                ]
                
                for pattern, trans_type in transition_patterns:
                    matches = re.finditer(pattern, etymology_text)
                    for match in matches:
                        lang, word_form = match.groups()
                        transitions.append(LanguageTransition(
                            from_lang=lang.strip(),
                            to_lang="English",
                            from_word=word_form.strip(),
                            to_word=word,
                            notes=f"{trans_type} relationship"
                        ))
                
                return EtymologyResult(
                    word=word,
                    earliest_ancestors=ancestors,
                    transitions=transitions,
                    sources=[EtymologySource(
                        name=self.source_name,
                        url=f"https://www.etymonline.com/word/{word}",
                        accessed_date=datetime.now().isoformat(),
                        confidence=self.confidence_score
                    )],
                    confidence_score=self.confidence_score
                )
                
        except Exception as e:
            logger.error(f"Web scraping failed for {word}: {str(e)}")
            return None 
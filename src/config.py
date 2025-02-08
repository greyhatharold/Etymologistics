"""
Global configuration settings for the Etymologistics project.

This module contains all configuration constants and settings used throughout the application.
All sensitive information (API keys, credentials) should be loaded from environment variables,
not stored directly in this file.
"""

import os
from pathlib import Path
from typing import Dict, List, Set

# Application Mode
DEBUG_MODE = os.getenv("ETYMOLOGISTICS_DEBUG", "false").lower() == "true"

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
VECTOR_DB_DIR = BASE_DIR / "data/vector_store"

# Language Family Configuration
class LanguageFamily:
    """
    Represents a language family with its constituent languages and metadata.
    
    Attributes:
        name (str): The name of the language family
        languages (Set[str]): Set of ISO 639-1 language codes in this family
        description (str): Brief description of the language family
    """
    def __init__(self, name: str, languages: Set[str], description: str):
        self.name = name
        self.languages = languages
        self.description = description

LANGUAGE_FAMILIES = {
    "indo-european": LanguageFamily(
        name="Indo-European",
        languages={
            # Germanic
            "en",  # English
            "de",  # German 
            "nl",  # Dutch
            "da",  # Danish
            "sv",  # Swedish
            "no",  # Norwegian
            "is",  # Icelandic
            "fy",  # Frisian
            "yi",  # Yiddish
            "got", # Gothic (extinct)
            "ang", # Old English
            "goh", # Old High German
            
            # Romance
            "fr",  # French
            "es",  # Spanish
            "it",  # Italian
            "pt",  # Portuguese
            "ro",  # Romanian
            "ca",  # Catalan
            "oc",  # Occitan
            "rm",  # Romansh
            "la",  # Latin
            "pro", # Old Provençal
            "fro", # Old French
            
            # Indo-Iranian
            "hi",  # Hindi
            "bn",  # Bengali
            "fa",  # Persian
            "ku",  # Kurdish
            "pa",  # Punjabi
            "ur",  # Urdu
            "sd",  # Sindhi
            "ne",  # Nepali
            "si",  # Sinhala
            "sa",  # Sanskrit
            "ae",  # Avestan
            "pli", # Pali
            
            # Slavic
            "ru",  # Russian
            "pl",  # Polish
            "cs",  # Czech
            "bg",  # Bulgarian
            "uk",  # Ukrainian
            "sr",  # Serbian
            "hr",  # Croatian
            "sl",  # Slovenian
            "sk",  # Slovak
            "mk",  # Macedonian
            "chu", # Old Church Slavonic
            "orv", # Old East Slavic
            
            # Hellenic
            "el",  # Greek (Modern)
            "grc", # Greek (Ancient)
            "gmy", # Mycenaean Greek
            
            # Baltic
            "lt",  # Lithuanian
            "lv",  # Latvian
            "prg", # Prussian (extinct)
            "sgs", # Samogitian
            
            # Celtic
            "ga",  # Irish
            "cy",  # Welsh
            "gd",  # Scottish Gaelic
            "br",  # Breton
            "kw",  # Cornish
            "gv",  # Manx
            "sga", # Old Irish
            "wlm", # Middle Welsh
            
            # Albanian
            "sq",  # Albanian
            "aln", # Gheg Albanian
            "als", # Tosk Albanian
            
            # Armenian
            "hy",  # Armenian
            "xcl", # Classical Armenian
            
            # Anatolian (extinct)
            "hit", # Hittite
            "xlu", # Luwian
            "xly", # Lycian
            "xlc", # Lydian
            
            # Tocharian (extinct) 
            "xto", # Tocharian A
            "txb", # Tocharian B
            
            # Indo-European isolates/minor branches
            "phr", # Phrygian (extinct)
            "xve", # Venetic (extinct)
            "xum", # Umbrian (extinct)
            "xos", # Oscan (extinct)
            "xtg", # Thracian (extinct)
            "ba",  # Basque
        },
        description="The largest language family, including most European and many South Asian languages"
    ),
    # Easy to add more families as needed:
    # "afroasiatic": LanguageFamily(...),
    # "sino-tibetan": LanguageFamily(...),
}

# External API Configuration
class APIEndpoint:
    """
    Configuration for external API endpoints.
    
    Attributes:
        base_url (str): Base URL for the API
        require_key (bool): Whether this API requires authentication
        env_key_name (str): Name of environment variable containing the API key
    """
    def __init__(self, base_url: str, require_key: bool, env_key_name: str = None):
        self.base_url = base_url
        self.require_key = require_key
        self.env_key_name = env_key_name

API_ENDPOINTS = {
    "wiktionary": APIEndpoint(
        base_url="https://en.wiktionary.org/w/api.php",
        require_key=False
    ),
    "etymonline": APIEndpoint(
        base_url="https://www.etymonline.com/api/v1",
        require_key=True,
        env_key_name="ETYMONLINE_API_KEY"
    ),
}

# Vector Database Configuration
VECTOR_DB_CONFIG = {
    "etymology_collection": "etymologies",  # Collection for etymology documents
    "stems_collection": "stems",           # Collection for stem analysis
    "embedding_dim": 384,                  # Default embedding dimension
    "similarity_metric": "cosine",         # Default similarity metric
    "min_similarity": 0.7,                 # Minimum similarity threshold
    "max_results": 10                      # Maximum number of similar results
}

# Web Scraping Configuration
SCRAPING_CONFIG = {
    "request_delay": 1.0,  # Delay between requests in seconds
    "timeout": 10,         # Request timeout in seconds
    "max_retries": 3,      # Maximum number of retry attempts
    "user_agent": "Etymologistics/1.0 (Research Project)"
}

# Cache Configuration
CACHE_CONFIG = {
    "max_size": "1GB",
    "ttl": 86400,  # 24 hours in seconds
    "compression": True
}

# Model Configuration
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "auto",  # "cpu", "cuda", or "auto"
    "batch_size": 32
}

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    "rotation": "500 MB",
    "retention": "10 days"
}

# API Configuration
API_CONFIG = {
    "timeout": 30,  # seconds
    "max_retries": 3,
    "retry_delay": 1  # seconds
} 
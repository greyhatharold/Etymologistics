"""
Browser automation utilities using browser-use.

This module provides a high-level browser automation interface for web scraping.
"""

import asyncio
import os
from typing import Optional, List
from loguru import logger
from browser_use import Agent
from playwright.async_api import Page
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Browser:
    """Browser automation wrapper using browser-use."""
    
    def __init__(self):
        """Initialize browser instance."""
        self._agent: Optional[Agent] = None
        self._pages: List[Page] = []
        self._llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self._initialized = False
        logger.info("Browser instance initialized")

    async def launch(self):
        """Launch the browser instance."""
        if not self._agent and not self._initialized:
            try:
                self._agent = Agent(
                    task="Navigate to etymonline.com and extract etymology information",
                    llm=self._llm,
                    browser_context={
                        "headless": True,
                        "viewport": {"width": 1280, "height": 720}
                    }
                )
                await self._agent.initialize()  # Initialize agent
                await self._agent.start()  # Start agent
                self._initialized = True
                logger.info("Browser launched")
            except Exception as e:
                logger.error(f"Failed to launch browser: {str(e)}")
                self._agent = None
                self._initialized = False
                raise

    async def create_page(self) -> Page:
        """
        Create a new browser page.
        
        Returns:
            Page: The created page instance
        """
        if not self._initialized:
            await self.launch()
        
        if not self._agent or not self._agent.browser:
            raise RuntimeError("Browser not properly initialized")
        
        page = await self._agent.browser.new_page()
        self._pages.append(page)
        return page

    async def close(self):
        """Close the browser and cleanup resources."""
        try:
            # Close all pages
            for page in self._pages:
                try:
                    await page.close()
                except Exception as e:
                    logger.warning(f"Error closing page: {str(e)}")
            self._pages = []
            
            # Close browser
            if self._agent and self._initialized:
                await self._agent.stop()
                self._agent = None
                self._initialized = False
                
            logger.info("Browser closed and resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during browser cleanup: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.launch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 
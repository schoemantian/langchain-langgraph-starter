#!/usr/bin/env python3

import os
import logging
from typing import Optional, Dict, Any, List
from getpass import getpass
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilySearchHandler:
    """Handles web search functionality using Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._setup_api_key()
        self.client = TavilyClient(api_key=self.api_key)
        logger.info("TavilySearchHandler initialized successfully")
    
    def _setup_api_key(self) -> str:
        """Setup Tavily API key from environment or user input."""
        if not os.environ.get("TAVILY_API_KEY"):
            api_key = getpass("Tavily API Key: ")
            os.environ["TAVILY_API_KEY"] = api_key
            logger.info("Tavily API key set successfully")
            return api_key
        return os.environ["TAVILY_API_KEY"]
    
    def search(self, query: str, max_results: int = 5, search_depth: str = "advanced") -> Dict[str, Any]:
        """
        Perform a web search using Tavily.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            search_depth: Search depth - "basic" or "advanced" (default: "advanced")
            
        Returns:
            Dictionary containing search results
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )
            logger.info(f"Search completed for query: {query}")
            return response
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e), "results": []}
    
    def search_news(self, query: str, days: int = 7, max_results: int = 3) -> str:
        """
        Search for recent news articles on a specific topic.
        
        Args:
            query: The search query
            days: How many days back to search (default: 7)
            max_results: Maximum number of results to return (default: 3)
            
        Returns:
            Formatted string of search results
        """
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                days=days,
                topic="news"
            )
            
            results = []
            for i, item in enumerate(response.get('results', []), 1):
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', 'No URL')
                
                results.append(f"**{i}. {title}**")
                results.append(f"{content[:200]}...")
                results.append(f"🔗 **FULL URL:** {url}\n")
            
            return "\n".join(results) if results else "No recent news found."
            
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return f"Search error: {str(e)}"
    
    def get_search_context(self, query: str) -> str:
        """
        Get search context for RAG applications.
        
        Args:
            query: Search query string
            
        Returns:
            Context string for RAG applications
        """
        try:
            context = self.client.get_search_context(query=query)
            logger.info(f"Search context generated for query: {query}")
            return context
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return f"Context search error: {str(e)}"
    
    def qna_search(self, query: str) -> str:
        """
        Get a quick answer to a question.
        
        Args:
            query: Question to answer
            
        Returns:
            Answer string
        """
        try:
            answer = self.client.qna_search(query=query)
            logger.info(f"Q&A search completed for query: {query}")
            return answer
        except Exception as e:
            logger.error(f"Q&A search failed: {e}")
            return f"Q&A search error: {str(e)}"


# LangChain Tools for integration with agents
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information on any topic. Always includes source URLs.
    
    Args:
        query: The search query to look up
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results with numbered sources and URLs
    """
    try:
        handler = TavilySearchHandler()
        response = handler.search(query, max_results)
        
        if "error" in response:
            return f"Search error: {response['error']}"
        
        results = []
        for i, item in enumerate(response.get('results', []), 1):
            title = item.get('title', 'No title')
            content = item.get('content', 'No content')
            url = item.get('url', 'No URL')
            
            results.append(f"**{i}. {title}**")
            results.append(f"{content[:300]}...")
            results.append(f"🔗 **FULL URL:** {url}\n")
        
        return "\n".join(results) if results else "No results found."
        
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool
def search_news(query: str, days: int = 7) -> str:
    """
    Search for recent news articles on a specific topic. Always includes source URLs.
    
    Args:
        query: The search query
        days: How many days back to search (default: 7)
        
    Returns:
        Formatted news results with numbered sources and URLs
    """
    try:
        handler = TavilySearchHandler()
        return handler.search_news(query, days)
    except Exception as e:
        return f"News search error: {str(e)}"


@tool
def get_current_info(query: str) -> str:
    """
    Get current, up-to-date information on any topic using web search.
    
    Args:
        query: The topic to get current information about
        
    Returns:
        Current information with sources
    """
    try:
        handler = TavilySearchHandler()
        return handler.get_search_context(query)
    except Exception as e:
        return f"Current info search error: {str(e)}"


@tool
def quick_answer(question: str) -> str:
    """
    Get a quick, direct answer to a question.
    
    Args:
        question: The question to answer
        
    Returns:
        Direct answer to the question
    """
    try:
        handler = TavilySearchHandler()
        return handler.qna_search(question)
    except Exception as e:
        return f"Quick answer error: {str(e)}"


# Get all available tools
def get_search_tools() -> List:
    """Get all available search tools for use with LangGraph agents."""
    return [web_search, search_news, get_current_info, quick_answer]


def main():
    """Demo function to test Tavily search functionality."""
    print("🔍 Tavily Search Demo")
    print("=" * 50)
    
    try:
        handler = TavilySearchHandler()
        
        # Test basic search
        print("\n1. Basic Web Search:")
        response = handler.search("latest AI developments 2024", max_results=3)
        if "error" not in response:
            for i, item in enumerate(response.get('results', [])[:2], 1):
                print(f"{i}. {item.get('title', 'No title')}")
                print(f"   {item.get('content', 'No content')[:150]}...")
                print(f"   Source: {item.get('url', 'No URL')}\n")
        
        # Test news search
        print("\n2. News Search:")
        news_results = handler.search_news("renewable energy investments", days=30)
        print(news_results[:300] + "..." if len(news_results) > 300 else news_results)
        
        # Test Q&A search
        print("\n3. Q&A Search:")
        answer = handler.qna_search("What is the current status of quantum computing?")
        print(answer)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have set your TAVILY_API_KEY environment variable")


if __name__ == "__main__":
    main()

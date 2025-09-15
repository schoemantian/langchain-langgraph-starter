import os
import logging
from getpass import getpass
from typing import Optional, Dict, Any, Generator, List
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from tavily_search import get_search_tools

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicAgent:
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0, enable_web_search: bool = False):
        self.model_name = model_name
        self.temperature = temperature
        self.enable_web_search = enable_web_search
        self.model = None
        self.agent = None
        self.memory = None 
        self.tools = []
        self._setup_api_key()
        self._initialize_model()
        self._setup_tools()
    
    def _setup_api_key(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            api_key = getpass("Google API Key: ")
            os.environ["GOOGLE_API_KEY"] = api_key
            logger.info("Google API key set successfully")
    
    def _setup_tools(self):
        """Setup tools for the agent."""
        if self.enable_web_search:
            try:
                self.tools = get_search_tools()
                logger.info(f"Web search tools loaded: {len(self.tools)} tools available")
            except Exception as e:
                logger.warning(f"Failed to load web search tools: {e}")
                self.tools = []
        else:
            self.tools = []
            logger.info("Web search disabled - no tools loaded")
    
    def _initialize_model(self):
        try:
            if ":" in self.model_name:
                model_name = self.model_name.split(":")[1]
            else:
                model_name = self.model_name
                
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature
            )
            logger.info(f"Model {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def create_simple_agent(self, prompt: str = "You are a helpful assistant."):
        try:
            if self.enable_web_search and self.tools:
                prompt = "You are a helpful assistant with access to web search. When users ask for current information, search for recent data and provide accurate, up-to-date responses. When you use web search tools, you MUST include the full URLs from the search results in your response. Do not just mention source names - include the complete URLs."
            
            self.agent = create_react_agent(
                self.model, 
                tools=self.tools, 
                prompt=prompt
            )
            logger.info("Simple agent created successfully")
        except Exception as e:
            logger.error(f"Failed to create simple agent: {e}")
            raise
    
    def create_agent_with_memory(self, prompt: str = "You are a helpful assistant. Remember our conversation history."):
        try:
            if self.enable_web_search and self.tools:
                prompt = "You are a helpful assistant with access to web search and memory. When users ask for current information, search for recent data and provide accurate, up-to-date responses. When you use web search tools, you MUST include the full URLs from the search results in your response. Do not just mention source names - include the complete URLs. Remember our conversation history."
            
            self.memory = MemorySaver()
            self.agent = create_react_agent(
                self.model, 
                tools=self.tools, 
                checkpointer=self.memory,
                prompt=prompt
            )
            logger.info("Agent with memory created successfully")
        except Exception as e:
            logger.error(f"Failed to create agent with memory: {e}")
            raise
    
    def safe_invoke(self, message: str, config: Optional[Dict[str, Any]] = None) -> str:
        if not self.agent:
            return "❌ Agent not initialized. Please create an agent first."
        
        try:
            response = self.agent.invoke({
                "messages": [{"role": "user", "content": message}]
            }, config or {})
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"❌ I apologize, but I encountered an error: {str(e)}"
    
    def safe_stream(self, message: str, config: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        if not self.agent:
            yield "❌ Agent not initialized. Please create an agent first."
            return
        
        try:
            for chunk in self.agent.stream({
                "messages": [{"role": "user", "content": message}]
            }, config or {}):
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        if hasattr(msg, 'content') and msg.content:
                            yield msg.content
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"❌ Streaming error: {str(e)}"
    
    def chat_with_memory(self, message: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        return self.safe_invoke(message, config)
    
    def stream_with_memory(self, message: str, thread_id: str = "default") -> Generator[str, None, None]:
        config = {"configurable": {"thread_id": thread_id}}
        yield from self.safe_stream(message, config)
    
    def toggle_web_search(self, enable: bool = None):
        """Toggle web search functionality on/off."""
        if enable is None:
            self.enable_web_search = not self.enable_web_search
        else:
            self.enable_web_search = enable
        
        self._setup_tools()
        
        if self.agent:
            if self.memory:
                self.create_agent_with_memory()
            else:
                self.create_simple_agent()
        
        status = "enabled" if self.enable_web_search else "disabled"
        logger.info(f"Web search {status}")
        return self.enable_web_search
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "web_search_enabled": self.enable_web_search,
            "tools_loaded": len(self.tools),
            "memory_enabled": self.memory is not None,
            "agent_initialized": self.agent is not None,
            "available_tools": [tool.name for tool in self.tools] if self.tools else []
        }


def main():
    print("🤖 Basic Agent with Web Search Demo")
    print("=" * 50)
    
    try:
        print("\n1. Basic Agent (no web search):")
        agent = BasicAgent()
        agent.create_simple_agent()
        response = agent.safe_invoke("Hello! Explain AI in one sentence.")
        print(f"Agent: {response}")
        print("\n2. Agent with Web Search:")
        agent_with_search = BasicAgent(enable_web_search=True)
        agent_with_search.create_agent_with_memory()
        
        response1 = agent_with_search.chat_with_memory("What are the latest developments in AI this week?", "user_123")
        print(f"Agent: {response1}")
        
        response2 = agent_with_search.chat_with_memory("Find recent news about renewable energy investments", "user_123")
        print(f"Agent: {response2}")
        
        print("\n3. Agent Status:")
        status = agent_with_search.get_agent_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n4. Error Handling Test:")
        response3 = agent_with_search.safe_invoke("Tell me about machine learning")
        print(f"Agent: {response3}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have set your GOOGLE_API_KEY and TAVILY_API_KEY environment variables")


if __name__ == "__main__":
    main()
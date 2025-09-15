import os
import logging
from getpass import getpass
from typing import Optional, Dict, Any, Generator
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicAgent:
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.model = None
        self.agent = None
        self.memory = None 
        self._setup_api_key()
        self._initialize_model()
    
    def _setup_api_key(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            api_key = getpass("Google API Key: ")
            os.environ["GOOGLE_API_KEY"] = api_key
            logger.info("Google API key set successfully")
    
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
            self.agent = create_react_agent(
                self.model, 
                tools=[], 
                prompt=prompt
            )
            logger.info("Simple agent created successfully")
        except Exception as e:
            logger.error(f"Failed to create simple agent: {e}")
            raise
    
    def create_agent_with_memory(self, prompt: str = "You are a helpful assistant. Remember our conversation history."):
        try:
            self.memory = MemorySaver()
            self.agent = create_react_agent(
                self.model, 
                tools=[], 
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


def main():
    print("🤖 Basic Agent Demo")
    print("=" * 50)
    
    try:
        agent = BasicAgent()
        
        print("\n1. Simple Agent (no memory):")
        agent.create_simple_agent()
        response = agent.safe_invoke("Hello! Explain AI in one sentence.")
        print(f"Agent: {response}")
        
        print("\n2. Agent with Memory:")
        agent.create_agent_with_memory()
        
        response1 = agent.chat_with_memory("My name is Alice and I love Python.", "user_123")
        print(f"Agent: {response1}")
        
        response2 = agent.chat_with_memory("What programming topics should I learn next?", "user_123")
        print(f"Agent: {response2}")
        
        print("\n3. Error Handling Test:")
        response3 = agent.safe_invoke("Tell me about machine learning")
        print(f"Agent: {response3}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have set your GOOGLE_API_KEY environment variable")


if __name__ == "__main__":
    main()
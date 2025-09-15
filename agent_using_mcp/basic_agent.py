import os
import logging
from getpass import getpass
from typing import Optional, Dict, Any, Generator
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from mcp_manager import MCPManager, MCPResponse

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
        self.mcp_manager = MCPManager()
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
    
    def _process_with_mcp(self, message: str) -> Optional[str]:
        """Process message through MCP system if applicable"""
        try:
            mcp_result = self.mcp_manager.execute_query(message)
            
            if mcp_result.success:
                data = mcp_result.data
                
                if isinstance(data, dict):
                    if data.get("type") == "random" or data.get("type") == "specific":
                        return f"🎭 Here's a dad joke for you:\n\n{data['joke']}\n\n(ID: {data['id']})"
                    
                    elif data.get("type") == "search":
                        jokes = data["jokes"][:3]
                        result = f"🔍 Found {data['total']} jokes for '{data['search_term']}':\n\n"
                        for i, joke in enumerate(jokes, 1):
                            result += f"{i}. {joke['joke']}\n\n"
                        return result
                    
                    elif data.get("type") == "evaluation":
                        return f"🧮 Mathematical Result:\n\nExpression: {data['expression']}\nResult: {data['result']}"
                
                return f"✅ MCP Result: {str(data)}"
            
            return None
            
        except Exception as e:
            logger.error(f"MCP processing error: {e}")
            return None
    
    def safe_invoke(self, message: str, config: Optional[Dict[str, Any]] = None) -> str:
        if not self.agent:
            return "❌ Agent not initialized. Please create an agent first."
        
        mcp_response = self._process_with_mcp(message)
        if mcp_response:
            return mcp_response
        
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
        
        mcp_response = self._process_with_mcp(message)
        if mcp_response:
            yield mcp_response
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
    
    def list_mcps(self) -> Dict[str, str]:
        """List available MCPs"""
        return self.mcp_manager.list_mcps()
    
    def toggle_mcp(self, mcp_name: str) -> bool:
        """Toggle MCP on/off"""
        return self.mcp_manager.toggle_mcp(mcp_name)
    
    def get_mcp_status(self) -> Dict[str, bool]:
        """Get status of all MCPs"""
        all_mcps = self.mcp_manager.mcps.keys()
        active_mcps = self.mcp_manager.active_mcps
        return {mcp: mcp in active_mcps for mcp in all_mcps}


def main():
    print("🤖 Basic Agent with MCP Demo")
    print("=" * 50)
    
    try:
        agent = BasicAgent()
        
        print("\n1. Simple Agent (no memory):")
        agent.create_simple_agent()
        response = agent.safe_invoke("Hello! Explain AI in one sentence.")
        print(f"Agent: {response}")
        
        print("\n2. MCP Testing:")
        
        joke_response = agent.safe_invoke("Tell me a dad joke")
        print(f"Joke MCP: {joke_response}")
        
        math_response = agent.safe_invoke("Calculate 25 * 4 + 10")
        print(f"Math MCP: {math_response}")
        
        specific_joke = agent.safe_invoke("@joke search for hipster jokes")
        print(f"Specific Joke MCP: {specific_joke}")
        
        print("\n3. Agent with Memory:")
        agent.create_agent_with_memory()
        
        response1 = agent.chat_with_memory("My name is Alice and I love Python.", "user_123")
        print(f"Agent: {response1}")
        
        response2 = agent.chat_with_memory("What programming topics should I learn next?", "user_123")
        print(f"Agent: {response2}")
        
        print("\n4. MCP Status:")
        mcps = agent.list_mcps()
        for name, desc in mcps.items():
            print(f"  {name}: {desc}")
        
        status = agent.get_mcp_status()
        print(f"MCP Status: {status}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have set your GOOGLE_API_KEY environment variable")


if __name__ == "__main__":
    main()
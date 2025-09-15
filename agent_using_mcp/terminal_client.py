#!/usr/bin/env python3
import os
import sys
from getpass import getpass
from basic_agent import BasicAgent


class TerminalClient:
    def __init__(self):
        self.agent = None
        self.memory_enabled = True
        self.thread_id = "terminal_session"
        self.setup_complete = False
        
    def setup_auth(self):
        """Handle authentication setup"""
        print("🔐 Authentication Setup")
        print("=" * 30)
        
        if not os.environ.get("GOOGLE_API_KEY"):
            print("Google API Key is required to use the agent.")
            api_key = getpass("Please enter your Google API Key: ")
            if not api_key.strip():
                print("❌ API Key is required. Exiting...")
                sys.exit(1)
            os.environ["GOOGLE_API_KEY"] = api_key
            print("✅ API Key set successfully!")
        else:
            print("✅ API Key already configured!")
        
        self.setup_complete = True
    
    def initialize_agent(self):
        """Initialize the agent after authentication"""
        try:
            print("\n🤖 Initializing Agent...")
            self.agent = BasicAgent()
            self.agent.create_agent_with_memory()
            print("✅ Agent initialized with memory enabled!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return False
    
    def display_help(self):
        """Display help information"""
        help_text = """
🤖 Basic Agent Terminal Client - Help
=====================================

Commands:
  /help          - Show this help message
  /clear         - Clear conversation memory
  /memory        - Toggle memory on/off
  /mcps          - List available MCPs and their status
  /toggle <mcp>  - Toggle specific MCP on/off
  /quit or /exit - Exit the application

MCP Usage:
  @joke          - Use joke MCP specifically
  @eval          - Use evaluation MCP specifically
  
  Or use natural language:
  - "tell me a joke" (uses joke MCP)
  - "calculate 2+2" (uses eval MCP)
  - "search for funny jokes" (uses joke MCP with search)

Examples:
  > tell me a dad joke
  > @joke search for programming jokes
  > calculate the square root of 144
  > @eval 2 * pi * 5
  > what is 25 + 30 * 2
  
Memory:
  - Conversations are remembered across messages
  - Use /clear to reset conversation
  - Use /memory to toggle memory on/off
        """
        print(help_text)
    
    def display_mcps(self):
        """Display MCP information"""
        print("\n📋 Available MCPs:")
        print("=" * 30)
        
        mcps = self.agent.list_mcps()
        status = self.agent.get_mcp_status()
        
        for name, description in mcps.items():
            status_icon = "✅" if status[name] else "❌"
            print(f"{status_icon} {name}: {description}")
        
        print("\nUsage:")
        print("- Use @mcp_name for specific MCP")
        print("- Or use keywords that trigger MCPs automatically")
        print("- Use /toggle <mcp_name> to enable/disable")
    
    def toggle_mcp(self, mcp_name: str):
        """Toggle MCP on/off"""
        if self.agent.toggle_mcp(mcp_name):
            status = self.agent.get_mcp_status()
            state = "enabled" if status[mcp_name] else "disabled"
            print(f"✅ MCP '{mcp_name}' {state}")
        else:
            print(f"❌ Unknown MCP: {mcp_name}")
    
    def process_command(self, user_input: str) -> bool:
        """Process special commands. Returns True if command was handled."""
        command = user_input.lower().strip()
        
        if command in ['/quit', '/exit']:
            print("👋 Goodbye!")
            return True
        
        elif command == '/help':
            self.display_help()
            return False
        
        elif command == '/clear':
            import time
            self.thread_id = f"terminal_session_{int(time.time())}"
            print("✅ Conversation memory cleared!")
            return False
        
        elif command == '/memory':
            self.memory_enabled = not self.memory_enabled
            state = "enabled" if self.memory_enabled else "disabled"
            print(f"✅ Memory {state}")
            return False
        
        elif command == '/mcps':
            self.display_mcps()
            return False
        
        elif command.startswith('/toggle '):
            mcp_name = command.replace('/toggle ', '').strip()
            self.toggle_mcp(mcp_name)
            return False
        
        elif command.startswith('/'):
            print(f"❌ Unknown command: {command}")
            print("💡 Type /help for available commands")
            return False
        
        return False
    
    def run(self):
        """Main terminal client loop"""
        print("🤖 Basic Agent Terminal Client")
        print("=" * 35)
        
        self.setup_auth()
        
        if not self.initialize_agent():
            sys.exit(1)
        
        print("\n💡 Type /help for commands or start chatting!")
        print("💡 Use @joke or @eval for specific MCPs")
        print("-" * 35)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if self.process_command(user_input):
                    break
                
                if self.memory_enabled:
                    response = self.agent.chat_with_memory(user_input, self.thread_id)
                else:
                    response = self.agent.safe_invoke(user_input)
                
                print(f"\nAgent: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Try again or type /help for assistance")


def main():
    client = TerminalClient()
    client.run()


if __name__ == "__main__":
    main()
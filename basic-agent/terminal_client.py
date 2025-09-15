#!/usr/bin/env python3

import os
import sys
import time
import signal
import argparse
import threading
from typing import Optional
from basic_agent import BasicAgent


class TerminalClient:
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.agent = BasicAgent(model_name, temperature)
        self.thread_id = "terminal_session"
        self.agent_initialized = False
        self.streaming = False
        self.interrupted = False
        self.thinking = False
        self.thinking_thread = None
        
    def initialize_agent(self, use_memory: bool = True):
        try:
            if use_memory:
                self.agent.create_agent_with_memory()
                print("✅ Agent initialized with memory")
            else:
                self.agent.create_simple_agent()
                print("✅ Agent initialized without memory")
            
            self.agent_initialized = True
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            print("💡 Check your GOOGLE_API_KEY in .env file")
            sys.exit(1)
    
    def print_welcome(self):
        print("\n" + "="*70)
        print("🤖 Basic Agent Terminal Client - Enhanced Edition")
        print("="*70)
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /memory   - Toggle memory on/off")
        print("  /status   - Show system status")
        print("  /quit     - Exit the program")
        print("="*70)
        print("💡 Tips:")
        print("  - Use Ctrl+C to interrupt long responses")
        print("  - Responses stream in real-time")
        print("  - Memory maintains conversation context")
        print("="*70)
        print("Start chatting! (Type your message and press Enter)")
        print("="*70 + "\n")
    
    def print_help(self):
        print("\n📖 Help & Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history (restart agent)")
        print("  /memory   - Toggle memory on/off (restart agent)")
        print("  /status   - Show current system status")
        print("  /quit     - Exit the program")
        print("\n💡 Features:")
        print("  - Real-time streaming responses")
        print("  - Memory for conversation context")
        print("  - Error handling and recovery")
        print("  - Interrupt long responses with Ctrl+C")
        print("  - Handle any query size or complexity")
        print()
    
    def show_status(self):
        print(f"\n🔧 System Status:")
        print(f"  Model: {self.agent.model_name}")
        print(f"  Temperature: {self.agent.temperature}")
        print(f"  Memory: {'Enabled' if self.agent.memory else 'Disabled'}")
        print(f"  Thread ID: {self.thread_id}")
        print(f"  Agent Status: {'Ready' if self.agent_initialized else 'Not Initialized'}")
        print(f"  Streaming: {'Active' if self.streaming else 'Inactive'}")
        print(f"  Thinking: {'Active' if self.thinking else 'Inactive'}")
        print()
    
    def clear_conversation(self):
        if self.agent_initialized:
            use_memory = self.agent.memory is not None
            self.initialize_agent(use_memory)
            print("🧹 Conversation history cleared")
        else:
            print("⚠️  Agent not initialized")
    
    def toggle_memory(self):
        if self.agent_initialized:
            use_memory = self.agent.memory is None
            self.initialize_agent(use_memory)
            status = "enabled" if use_memory else "disabled"
            print(f"🧠 Memory {status}")
        else:
            print("⚠️  Agent not initialized")
    
    def process_command(self, user_input: str) -> bool:
        command = user_input.strip().lower()
        
        if command == "/help":
            self.print_help()
            return True
        elif command == "/clear":
            self.clear_conversation()
            return True
        elif command == "/memory":
            self.toggle_memory()
            return True
        elif command == "/status":
            self.show_status()
            return True
        elif command == "/quit":
            print("👋 Goodbye!")
            return True
        else:
            return False
    
    def signal_handler(self, signum, frame):
        if self.streaming or self.thinking:
            print("\n⏹️  Response interrupted by user")
            self.interrupted = True
            self.streaming = False
            self.thinking = False
        else:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    def start_thinking_indicator(self):
        def thinking_animation():
            dots = ["", ".", "..", "..."]
            i = 0
            while self.thinking and not self.interrupted:
                print(f"\r🤖 Agent: Thinking{dots[i % len(dots)]}", end="", flush=True)
                i += 1
                time.sleep(0.5)
            if not self.interrupted:
                print(f"\r🤖 Agent: ", end="", flush=True)
        
        self.thinking = True
        self.thinking_thread = threading.Thread(target=thinking_animation, daemon=True)
        self.thinking_thread.start()
    
    def stop_thinking_indicator(self):
        self.thinking = False
        if self.thinking_thread:
            self.thinking_thread.join(timeout=0.1)
    
    def stream_agent_response(self, message: str) -> str:
        if not self.agent_initialized:
            return "❌ Agent not initialized. Please restart the program."
        
        try:
            self.streaming = True
            self.interrupted = False
            
            self.start_thinking_indicator()
            
            if self.agent.memory:
                config = {"configurable": {"thread_id": self.thread_id}}
                response = self.agent.agent.invoke({
                    "messages": [{"role": "user", "content": message}]
                }, config)
            else:
                response = self.agent.agent.invoke({
                    "messages": [{"role": "user", "content": message}]
                })
            
            self.stop_thinking_indicator()
            
            if self.interrupted:
                return "\n⏹️  Response interrupted"
            
            content = response["messages"][-1].content
            print(content)
            return content
            
        except KeyboardInterrupt:
            self.stop_thinking_indicator()
            print("\n⏹️  Response interrupted")
            return ""
        except Exception as e:
            self.stop_thinking_indicator()
            error_msg = f"❌ Error: {str(e)}"
            print(f"\n{error_msg}")
            print("💡 Troubleshooting:")
            print("  - Check your internet connection")
            print("  - Verify your GOOGLE_API_KEY is correct")
            print("  - Try /clear to reset the conversation")
            print("  - Check /status for system information")
            return error_msg
        finally:
            self.streaming = False
    
    def get_agent_response_safe(self, message: str) -> str:
        try:
            return self.stream_agent_response(message)
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    def run_interactive(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.print_welcome()
        self.initialize_agent(use_memory=True)
        
        try:
            while True:
                try:
                    if not self.thinking and not self.streaming:
                        user_input = input("\n👤 You: ").strip()
                    else:
                        print("\n⏳ Please wait, AI is thinking...")
                        time.sleep(1)
                        continue
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                if self.process_command(user_input):
                    continue
                
                try:
                    self.get_agent_response_safe(user_input)
                except Exception as e:
                    print(f"\n❌ Critical error: {e}")
                    print("💡 Try restarting the program")
                    break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Please restart the program")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Basic Agent Terminal Client")
    parser.add_argument(
        "--model", 
        default="gemini-2.5-flash",
        help="Model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Model temperature (default: 0.0)"
    )
    parser.add_argument(
        "--no-memory", 
        action="store_true",
        help="Disable memory for conversation context"
    )
    
    args = parser.parse_args()
    
    try:
        client = TerminalClient(args.model, args.temperature)
        
        if args.no_memory:
            client.initialize_agent(use_memory=False)
        else:
            client.initialize_agent(use_memory=True)
        
        client.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        print("💡 Check your setup and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
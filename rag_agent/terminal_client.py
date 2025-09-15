#!/usr/bin/env python3

import os
import sys
import time
import signal
import argparse
import threading
from dotenv import load_dotenv
from rag_agent import RAGAgent

load_dotenv()


class TerminalClient:
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("❌ GOOGLE_API_KEY environment variable not set. Please set it to use RAG functionality.")
            sys.exit(1)
        self.rag_agent = RAGAgent(
            google_api_key=google_api_key,
            model_name=model_name,
            knowledge_base_dir="knowledge_base",
            vector_store_path="vector_store"
        )
        
        self.thread_id = "terminal_session"
        self.agent_initialized = False
        self.streaming = False
        self.interrupted = False
        self.thinking = False
        self.thinking_thread = None
        
    def initialize_agent(self):
        try:
            self.rag_agent.create_vector_store()
            self.rag_agent.create_graph()
            print("✅ RAG Agent initialized with knowledge base")
            self.agent_initialized = True
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            print("💡 Check your GOOGLE_API_KEY environment variable")
            sys.exit(1)
    
    def print_welcome(self):
        print("\n" + "="*70)
        print("🤖 RAG Agent Terminal Client - Enhanced Edition")
        print("="*70)
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /status   - Show system status")
        print("  /stats    - Show knowledge base statistics")
        print("  /adddoc   - Add document to knowledge base")
        print("  /quit     - Exit the program")
        print("="*70)
        print("💡 Tips:")
        print("  - Use Ctrl+C to interrupt long responses")
        print("  - RAG agent uses knowledge base for context")
        print("  - Ask questions about the loaded documents")
        print("="*70)
        print("Start chatting! (Type your message and press Enter)")
        print("="*70 + "\n")
    
    def print_help(self):
        print("\n📖 Help & Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history (restart agent)")
        print("  /status   - Show current system status")
        print("  /stats    - Show knowledge base statistics")
        print("  /adddoc   - Add document to knowledge base")
        print("  /quit     - Exit the program")
        print("\n💡 Features:")
        print("  - RAG (Retrieval-Augmented Generation) responses")
        print("  - Knowledge base integration")
        print("  - Document-based context")
        print("  - Source citation")
        print("  - Error handling and recovery")
        print("  - Interrupt long responses with Ctrl+C")
        print("  - Handle any query size or complexity")
        print()
    
    def show_status(self):
        print("\n🔧 System Status:")
        print("  Agent Type: RAG Agent")
        print(f"  Model: {self.rag_agent.model_name}")
        print(f"  Embedding Model: {self.rag_agent.embedding_model}")
        print(f"  Knowledge Base: {self.rag_agent.knowledge_base_dir}")
        print(f"  Vector Store: {self.rag_agent.vector_store_path}")
        print(f"  Thread ID: {self.thread_id}")
        print(f"  Agent Status: {'Ready' if self.agent_initialized else 'Not Initialized'}")
        print(f"  Streaming: {'Active' if self.streaming else 'Inactive'}")
        print(f"  Thinking: {'Active' if self.thinking else 'Inactive'}")
        print()
    
    def clear_conversation(self):
        if self.agent_initialized:
            self.initialize_agent()
            print("🧹 RAG Agent reinitialized")
        else:
            print("⚠️  Agent not initialized")
    
    def show_stats(self):
        """Show knowledge base statistics for RAG agent."""
        if not self.agent_initialized:
            print("⚠️  Agent not initialized")
            return
        
        try:
            stats = self.rag_agent.get_stats()
            print("\n📊 Knowledge Base Statistics:")
            print(f"  Total Documents: {stats.get('total_documents', 0)}")
            print(f"  Total Chunks: {stats.get('total_chunks', 0)}")
            print(f"  Chunk Size: {stats.get('chunk_size', 'N/A')}")
            print(f"  Chunk Overlap: {stats.get('chunk_overlap', 'N/A')}")
            print(f"  Top K: {stats.get('top_k', 'N/A')}")
            print(f"  Model: {stats.get('model_name', 'N/A')}")
            print(f"  Embedding Model: {stats.get('embedding_model', 'N/A')}")
            
            files = stats.get('files', [])
            if files:
                print("\n📁 Files in Knowledge Base:")
                for i, file_path in enumerate(files, 1):
                    filename = os.path.basename(file_path)
                    print(f"  {i}. {filename}")
            print()
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def add_document(self):
        """Add a document to the knowledge base."""
        if not self.agent_initialized:
            print("⚠️  Agent not initialized")
            return
        
        try:
            file_path = input("Enter path to document: ").strip()
            if not file_path:
                print("❌ No file path provided")
                return
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return
            
            self.rag_agent.add_document(file_path)
            print(f"✅ Document added successfully: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error adding document: {e}")
    
    def process_command(self, user_input: str) -> bool:
        command = user_input.strip().lower()
        
        if command == "/help":
            self.print_help()
            return True
        elif command == "/clear":
            self.clear_conversation()
            return True
        elif command == "/status":
            self.show_status()
            return True
        elif command == "/stats":
            self.show_stats()
            return True
        elif command == "/adddoc":
            self.add_document()
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
                print("\r🤖 Agent: ", end="", flush=True)
        
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
            
            result = self.rag_agent.query(message, self.thread_id)
            response = result["response"]
            sources = result.get("sources", [])
            
            self.stop_thinking_indicator()
            
            if self.interrupted:
                return "\n⏹️  Response interrupted"
            
            print(response)
            
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    filename = os.path.basename(source.get("source", "Unknown"))
                    print(f"  {i}. {filename}")
                    if source.get("content"):
                        content_preview = source["content"][:100] + "..." if len(source["content"]) > 100 else source["content"]
                        print(f"     {content_preview}")
            
            return response
            
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
        self.initialize_agent()
        
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
    parser = argparse.ArgumentParser(description="RAG Agent Terminal Client")
    parser.add_argument(
        "--model", 
        default="gemini-1.5-flash",
        help="Model to use (default: gemini-1.5-flash)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Model temperature (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    try:
        client = TerminalClient(args.model, args.temperature)
        client.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        print("💡 Check your setup and try again")
        print("💡 Make sure GOOGLE_API_KEY is set")
        sys.exit(1)


if __name__ == "__main__":
    main()
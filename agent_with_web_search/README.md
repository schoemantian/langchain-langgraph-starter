# Step 2: Agent with Web Search - LangChain and LangGraph

A powerful chat agent using Google Gemini with memory, web search capabilities, and terminal interface. This agent can access real-time information through Tavily web search integration.

> **This is Step 2** - Building upon the basic agent from Step 1 with enhanced web search capabilities.

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Packages
```bash
pip install -r requirements.txt
```

### 3. Set API Keys
```bash
cp example.env .env
# Edit .env and add your API keys
```

Required API Keys:
- `GOOGLE_API_KEY` - For Google Gemini model access
- `TAVILY_API_KEY` - For web search functionality (get from [Tavily](https://tavily.com))

## Usage

### Run Terminal Client

Basic usage:
```bash
python terminal_client.py
```

With web search enabled:
```bash
python terminal_client.py --web-search
```

With custom model and temperature:
```bash
python terminal_client.py --model gemini-2.5-flash --temperature 0.1 --web-search
```

### Terminal Commands
- `/help` - Show help and available commands
- `/clear` - Clear conversation history
- `/memory` - Toggle memory on/off
- `/search` - Toggle web search on/off
- `/status` - Show system status and configuration
- `/debug` - Toggle debug mode
- `/quit` - Exit the program

### Memory Management
- Memory is enabled by default
- Each conversation thread is separate
- Use `/memory` to toggle on/off
- Use `/clear` to reset conversation

### Web Search Features
- **Real-time Information**: Access current events and news
- **Up-to-date Data**: Get information beyond training cutoff
- **Source Attribution**: Always know where information comes from
- **Multiple Search Types**: General search, news search, Q&A, and context generation
- **Toggle On/Off**: Use `/search` command to enable/disable web search

### Basic Usage

#### Without Web Search
```python
from basic_agent import BasicAgent

agent = BasicAgent()
agent.create_agent_with_memory()
response = agent.chat_with_memory("Hello!", "user_123")
print(response)
```

#### With Web Search
```python
from basic_agent import BasicAgent

# Enable web search
agent = BasicAgent(enable_web_search=True)
agent.create_agent_with_memory()

# Ask for current information
response = agent.chat_with_memory("What's the latest news about AI?", "user_123")
print(response)
```

#### Using Search Tools Directly
```python
from tavily_search import web_search, search_news, get_current_info

# General web search
results = web_search("latest developments in quantum computing")
print(results)

# News search
news = search_news("renewable energy investments", days=7)
print(news)

# Get current context
context = get_current_info("artificial intelligence trends 2024")
print(context)
```

## What's New in Step 2

This step extends the basic agent from Step 1 with powerful web search capabilities:

### New Features Added
- **Web Search Integration**: Real-time information access via Tavily
- **Multiple Search Tools**: General search, news search, Q&A, and context generation
- **Source Attribution**: Always know where information comes from
- **Toggle Functionality**: Enable/disable web search without restarting
- **Enhanced Terminal Interface**: New `/search` command and improved status display

### Core Capabilities (from Step 1)
- **Memory Management**: Persistent conversation context
- **Streaming Responses**: Real-time response streaming
- **Error Handling**: Robust error recovery and user feedback
- **Interactive Terminal**: Rich command-line interface

### Web Search Tools
- `web_search`: General web search with source attribution
- `search_news`: Recent news articles on specific topics
- `get_current_info`: Context generation for RAG applications
- `quick_answer`: Direct Q&A responses

### Command Line Options
- `--model`: Choose the language model (default: gemini-2.5-flash)
- `--temperature`: Set model creativity (default: 0.0)
- `--no-memory`: Disable conversation memory
- `--web-search`: Enable web search capabilities

## Files

### Core Files (Enhanced from Step 1)
- `basic_agent.py` - Core agent with web search integration
- `terminal_client.py` - Enhanced CLI interface with web search commands

### New Files (Step 2)
- `tavily_search.py` - Web search functionality and tools
- `example.env` - API key template (updated with Tavily key)
- `requirements.txt` - Dependencies including Tavily integration

## Progression from Step 1

This step builds upon the foundation established in Step 1:

1. **Step 1** (`basic-agent/`): Basic agent with memory and terminal interface
2. **Step 2** (`agent_with_web_search/`): Enhanced agent with web search capabilities

Both steps maintain identical terminal interfaces, ensuring a consistent user experience while adding powerful new capabilities.
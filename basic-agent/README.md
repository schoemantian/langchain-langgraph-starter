# Basic Agent with LangChain and LangGraph

A simple chat agent using Google Gemini with memory and terminal interface.

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

### 3. Set API Key
```bash
cp example.env .env
# Edit .env and add your Google API key
```

## Usage

### Run Terminal Client
```bash
python terminal_client.py
```

### Terminal Commands
- `/help` - Show help
- `/clear` - Clear conversation
- `/memory` - Toggle memory on/off
- `/quit` - Exit

### Memory Management
- Memory is enabled by default
- Each conversation thread is separate
- Use `/memory` to toggle on/off
- Use `/clear` to reset conversation

### Basic Usage
```python
from basic_agent import BasicAgent

agent = BasicAgent()
agent.create_agent_with_memory()
response = agent.chat_with_memory("Hello!", "user_123")
print(response)
```

## Files
- `basic_agent.py` - Core agent
- `terminal_client.py` - CLI interface
- `example.env` - API key template
- `requirements.txt` - Dependencies
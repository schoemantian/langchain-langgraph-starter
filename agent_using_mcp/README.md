# Basic Agent with LangChain, LangGraph, and MCP

A sophisticated chat agent using Google Gemini with memory, terminal interface, and Model Context Protocol (MCP) integration for enhanced functionality.

## Features

- 🤖 **Google Gemini Integration** - Powered by Google's advanced AI model
- 🧠 **Memory Management** - Persistent conversation history
- 🔧 **MCP Support** - Modular Context Protocols for extended functionality
- 🎭 **Dad Jokes** - Integration with icanhazdadjoke.com API
- 🧮 **Math Evaluation** - Safe mathematical expression evaluation
- 💬 **Terminal Interface** - Interactive command-line client

## MCP Modules

### 1. Joke MCP (@joke)
- Fetches dad jokes from icanhazdadjoke.com
- Supports random, search, and specific joke retrieval
- Natural language triggers: "joke", "funny", "humor"

### 2. Evaluation MCP (@eval)
- Safe mathematical expression evaluation
- Supports math functions and constants
- Natural language triggers: "calculate", "math", "compute"

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

The terminal client will:
1. **Prompt for API key** if not set in environment
2. **Initialize the agent** with MCP modules
3. **Start interactive session** with memory enabled

### Terminal Commands
- `/help` - Show comprehensive help
- `/clear` - Clear conversation memory
- `/memory` - Toggle memory on/off
- `/mcps` - List available MCPs and status
- `/toggle <mcp>` - Toggle specific MCP on/off
- `/quit` or `/exit` - Exit application

### MCP Usage Examples

#### Joke MCP
```bash
> tell me a dad joke
> @joke search for programming jokes
> give me something funny
```

#### Evaluation MCP
```bash
> calculate 25 * 4 + 10
> @eval sqrt(144) + 2^3
> what is 2 * pi * 5
```

### Basic Usage
```python
from basic_agent import BasicAgent

agent = BasicAgent()
agent.create_agent_with_memory()

# MCP will automatically handle joke and math requests
response = agent.chat_with_memory("Tell me a joke", "user_123")
print(response)

# Use specific MCP
response = agent.chat_with_memory("@eval calculate 10 * 5", "user_123")
print(response)
```

## Files
- `basic_agent.py` - Core agent with MCP integration
- `mcp_manager.py` - MCP management and implementations
- `terminal_client.py` - Enhanced CLI interface
- `example.env` - API key template
- `requirements.txt` - Dependencies

## MCP Architecture

The MCP system provides a modular way to extend agent functionality:

1. **MCPManager** - Routes queries to appropriate MCPs
2. **BaseMCP** - Abstract base class for all MCPs
3. **JokeMCP** - Dad jokes from icanhazdadjoke.com
4. **EvalMCP** - Safe mathematical evaluation

### Adding New MCPs

1. Create a new class inheriting from `BaseMCP`
2. Implement `execute()` and `get_description()` methods
3. Register in `MCPManager.__init__()`
4. Add keyword detection in `parse_query()`

## Error Handling

- Graceful fallback to agent when MCPs fail
- Comprehensive error messages
- Safe mathematical evaluation with restricted environment
- Network timeout handling for API calls

## Memory Management

- Memory is enabled by default
- Each conversation thread is separate
- Use `/memory` to toggle on/off
- Use `/clear` to reset conversation
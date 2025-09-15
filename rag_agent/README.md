# RAG Agent with LangChain and LangGraph

A powerful Retrieval-Augmented Generation (RAG) agent using LangChain and LangGraph with Google Gemini integration for both language model and embeddings.

## Features

- **Document-based knowledge retrieval** - Query your knowledge base documents
- **Google Gemini integration** - Uses Gemini for both LLM and embeddings
- **Vector store with semantic search** - Powered by Gemini embeddings and FAISS
- **Source citation and transparency** - See which documents informed the response
- **Multiple document formats** - Supports TXT, MD, JSON, and CSV files
- **Real-time streaming responses** - Interactive terminal interface
- **Error handling and recovery** - Robust error management

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

### 3. Set Google API Key
Copy the example environment file and add your API key:

```bash
cp example.env .env
```

Then edit the `.env` file and replace the placeholder with your actual Google API key:
```env
GOOGLE_API_KEY=your-actual-google-api-key-here
```

**Note:** You can get a Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Set Up Knowledge Base (Optional)
The RAG agent works with documents in the `knowledge_base/` directory. You can add your own documents:

```bash
# Create knowledge base directory (if it doesn't exist)
mkdir -p knowledge_base

# Add your documents (supports TXT, MD, JSON, CSV)
cp your-document.txt knowledge_base/
cp your-notes.md knowledge_base/
cp your-data.json knowledge_base/
cp your-spreadsheet.csv knowledge_base/
```

**Supported file formats:**
- **TXT files** - Plain text documents
- **MD files** - Markdown documents  
- **JSON files** - Structured data
- **CSV files** - Spreadsheet data

## Usage

### 1. Create Embeddings and Vector Store
First, create embeddings from your knowledge base documents:

```bash
python embeddings_manager.py
```

This will:
- Process all documents in the `knowledge_base/` directory
- Create embeddings using Google Gemini's embedding model
- Build a vector store for semantic search
- Save metadata for future use

### 2. Run the Terminal Client
Start the interactive RAG agent:

```bash
python terminal_client.py
```

The terminal client will:
- Initialize the RAG agent with your knowledge base
- Load the vector store
- Provide an interactive chat interface
- Show source citations for responses

### 3. Terminal Commands
Once running, you can use these commands:
- `/help` - Show help message
- `/clear` - Clear conversation history
- `/status` - Show system status
- `/stats` - Show knowledge base statistics
- `/adddoc` - Add document to knowledge base
- `/quit` - Exit the program

## Troubleshooting

### Common Issues

**"API key not valid" error:**
- Make sure you've updated the `.env` file with your actual Google API key
- Verify the API key is correct and has the necessary permissions
- Check that the `.env` file is in the same directory as the scripts

**"No documents found in knowledge_base" warning:**
- This is normal if you haven't added any documents yet
- Add documents to the `knowledge_base/` directory and run `python embeddings_manager.py` again
- The agent will still work but won't have document context

**"Retriever not initialized" error:**
- Run `python embeddings_manager.py` first to create the vector store
- Make sure you have documents in the `knowledge_base/` directory

**Import errors:**
- Make sure you're in the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Getting Help

If you encounter issues:
1. Check the error messages for specific guidance
2. Verify your Google API key is valid
3. Ensure all dependencies are installed
4. Check that documents are in the correct format

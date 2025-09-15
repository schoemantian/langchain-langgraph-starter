import os
import json
import logging
from typing import List, Dict, Any, Optional, Annotated, Sequence
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader,
    JSONLoader,
    CSVLoader
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for the RAG agent containing messages and retrieved context."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[Document]
    question: str


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent using LangChain and LangGraph.
    """
    
    def __init__(self, 
                 google_api_key: str,
                 model_name: str = "gemini-1.5-flash",
                 embedding_model: str = "models/embedding-001",
                 knowledge_base_dir: str = "knowledge_base",
                 vector_store_path: str = "vector_store",
                 top_k: int = 5,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG Agent.
        
        Args:
            google_api_key: Google API key for Gemini
            model_name: Gemini model to use for generation
            embedding_model: Gemini embedding model
            knowledge_base_dir: Directory containing knowledge base files
            vector_store_path: Path to save/load vector store
            top_k: Number of top documents to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=google_api_key
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = None
        self.retriever = None
        self.graph = None
        
        os.makedirs(knowledge_base_dir, exist_ok=True)
        os.makedirs(vector_store_path, exist_ok=True)
        
        logger.info(f"RAG Agent initialized with model: {model_name}")
    
    def load_documents(self) -> List[Document]:
        """Load documents from the knowledge base directory."""
        documents = []
        
        text_loader = DirectoryLoader(
            self.knowledge_base_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents.extend(text_loader.load())
        
        md_loader = DirectoryLoader(
            self.knowledge_base_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents.extend(md_loader.load())
        
        json_files = list(Path(self.knowledge_base_dir).glob("**/*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={"source": str(json_file)}
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading JSON file {json_file}: {str(e)}")
        
        csv_loader = DirectoryLoader(
            self.knowledge_base_dir,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents.extend(csv_loader.load())
        
        logger.info(f"Loaded {len(documents)} documents from {self.knowledge_base_dir}")
        return documents
    
    def create_vector_store(self):
        """Create or load vector store from knowledge base."""
        try:
            if self.load_vector_store():
                logger.info("Loaded existing vector store")
                return
            
            logger.info("No existing vector store found, creating from documents...")
            documents = self.load_documents()
            
            if not documents:
                logger.warning(f"No documents found in {self.knowledge_base_dir}")
                return
            
            doc_splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(doc_splits)} chunks")
            
            self.vector_store = InMemoryVectorStore.from_documents(
                documents=doc_splits,
                embedding=self.embeddings
            )
            
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            
            self._save_vector_store_metadata(doc_splits)
            
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self):
        """Load existing vector store from disk."""
        try:
            import pickle
            
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            if not os.path.exists(metadata_path):
                logger.info("No existing vector store metadata found")
                return False
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            documents = []
            for doc_data in metadata.get('documents', []):
                doc = Document(
                    page_content=doc_data['page_content'],
                    metadata=doc_data['metadata']
                )
                documents.append(doc)
            
            if not documents:
                logger.warning("No documents found in loaded metadata")
                return False
            
            self.vector_store = InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            
            logger.info(f"Loaded vector store with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _save_vector_store_metadata(self, doc_splits: List[Document]):
        """Save metadata about the vector store."""
        metadata = {
            "total_documents": len(set(doc.metadata.get("source", "") for doc in doc_splits)),
            "total_chunks": len(doc_splits),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "created_at": datetime.now().isoformat(),
            "files": list(set(doc.metadata.get("source", "") for doc in doc_splits))
        }
        
        metadata_path = os.path.join(self.vector_store_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_vector_store_metadata(self) -> Dict[str, Any]:
        """Load vector store metadata."""
        metadata_path = os.path.join(self.vector_store_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents for the question."""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return state
        
        try:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                question = last_message.content
            else:
                question = state.get("question", "")
            
            retrieved_docs = self.retriever.invoke(question)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for question: {question}")
            
            return {
                **state,
                "context": retrieved_docs,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return state
    
    def generate_response(self, state: RAGState) -> RAGState:
        """Generate response using retrieved context."""
        try:
            context_docs = state.get("context", [])
            context = "\n\n".join([doc.page_content for doc in context_docs])
            question = state.get("question", "")
            
            prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided context.

Context information:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided above. 
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which documents you're referencing when possible.

Answer:
""")
            
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke({
                "context": context,
                "question": question
            })
            
            ai_message = AIMessage(content=response)
            
            return {
                **state,
                "messages": state["messages"] + [ai_message]
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_message = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
            return {
                **state,
                "messages": state["messages"] + [error_message]
            }
    
    def create_graph(self):
        """Create the LangGraph workflow."""
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_response)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        self.graph = workflow.compile()
        
        logger.info("RAG graph created successfully")
    
    def query(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Query the RAG agent with a question.
        
        Args:
            question: User question
            thread_id: Thread ID for conversation context
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.graph:
            self.create_graph()
        
        if not self.vector_store:
            self.create_vector_store()
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "context": [],
                "question": question
            }
            
            result = self.graph.invoke(
                initial_state,
                {"configurable": {"thread_id": thread_id}}
            )
            
            response = result["messages"][-1].content
            context_docs = result.get("context", [])
            
            sources = []
            for doc in context_docs:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A")
                })
            
            return {
                "response": response,
                "sources": sources,
                "num_sources": len(sources),
                "timestamp": datetime.now().isoformat(),
                "thread_id": thread_id
            }
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "timestamp": datetime.now().isoformat(),
                "thread_id": thread_id
            }
    
    def add_document(self, file_path: str):
        """Add a new document to the knowledge base and update vector store."""
        try:
            import shutil
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.knowledge_base_dir, filename)
            shutil.copy2(file_path, dest_path)
            
            self.create_vector_store()
            
            logger.info(f"Added document: {filename}")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG agent."""
        metadata = self.load_vector_store_metadata()
        
        return {
            "vector_store_initialized": self.vector_store is not None,
            "retriever_initialized": self.retriever is not None,
            "graph_initialized": self.graph is not None,
            "knowledge_base_dir": self.knowledge_base_dir,
            "vector_store_path": self.vector_store_path,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            **metadata
        }


def main():
    """Example usage of the RAG Agent."""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    agent = RAGAgent(
        google_api_key=GOOGLE_API_KEY,
        model_name="gemini-1.5-flash",
        knowledge_base_dir="knowledge_base",
        vector_store_path="vector_store"
    )
    
    agent.create_vector_store()
    
    stats = agent.get_stats()
    print("RAG Agent Statistics:")
    print(json.dumps(stats, indent=2))
    
    print("\nRAG Agent Ready! Type 'quit' to exit.")
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        result = agent.query(question)
        
        print(f"\nResponse: {result['response']}")
        print(f"\nSources ({result['num_sources']}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['source']}")
            print(f"   {source['content']}")
            print()


if __name__ == "__main__":
    main()

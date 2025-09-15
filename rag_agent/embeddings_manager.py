import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader,
    JSONLoader,
    CSVLoader,
    PyPDFLoader
)
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles processing of different document types."""
    
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """Read content from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    @staticmethod
    def read_json_file(file_path: str) -> str:
        """Read and format content from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    
    @staticmethod
    def read_csv_file(file_path: str) -> str:
        """Read and format content from CSV file."""
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()
    
    @staticmethod
    def read_pdf_file(file_path: str) -> str:
        """Read and extract text content from PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = "\n\n".join([page.page_content for page in pages])
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8: 
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks


class EmbeddingsManager:
    """Manages document embeddings and vector store operations using LangChain."""
    
    def __init__(self, 
                 google_api_key: str,
                 embedding_model: str = "models/embedding-001",
                 knowledge_base_dir: str = "knowledge_base",
                 vector_store_path: str = "vector_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embeddings manager.
        
        Args:
            google_api_key: Google API key for Gemini
            embedding_model: Name of the Gemini embedding model
            knowledge_base_dir: Directory containing documents
            vector_store_path: Directory to save vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.google_api_key = google_api_key
        self.embedding_model = embedding_model
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = None
        self.documents = []
        self.chunks = []
        
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        self.doc_processor = DocumentProcessor()
        
        logger.info(f"EmbeddingsManager initialized with embedding model: {embedding_model}")
    
    def _get_supported_files(self) -> List[str]:
        """Get list of supported file types in knowledge base directory."""
        supported_extensions = {'.txt', '.md', '.json', '.csv', '.pdf'}
        files = []
        
        for root, _, filenames in os.walk(self.knowledge_base_dir):
            for filename in filenames:
                if Path(filename).suffix.lower() in supported_extensions:
                    files.append(os.path.join(root, filename))
        
        return files
    
    def _read_document(self, file_path: str) -> str:
        """Read document content based on file type."""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext in ['.txt', '.md']:
                return self.doc_processor.read_text_file(file_path)
            elif file_ext == '.json':
                return self.doc_processor.read_json_file(file_path)
            elif file_ext == '.csv':
                return self.doc_processor.read_csv_file(file_path)
            elif file_ext == '.pdf':
                return self.doc_processor.read_pdf_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return ""
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""
    
    def _process_document(self, file_path: str) -> List[Document]:
        """
        Process a document into chunks with metadata.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects with metadata
        """
        content = self._read_document(file_path)
        if not content:
            return []
        
        chunks = self.doc_processor.chunk_text(
            content, 
            self.chunk_size, 
            self.chunk_overlap
        )
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'file_path': file_path,
                    'chunk_id': i,
                    'file_size': len(content),
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks),
                    'processed_at': datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        logger.info(f"Processed {file_path}: {len(chunks)} chunks")
        return documents
    
    def create_embeddings_from_directory(self):
        """Create embeddings for all supported files in the knowledge base directory."""
        logger.info("Creating embeddings from knowledge base directory...")
        
        files = self._get_supported_files()
        if not files:
            logger.warning(f"No supported files found in {self.knowledge_base_dir}")
            return
        
        all_documents = []
        
        for file_path in files:
            documents = self._process_document(file_path)
            all_documents.extend(documents)
        
        if not all_documents:
            logger.warning("No text content found to create embeddings")
            return
        
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=all_documents,
            embedding=self.embeddings
        )
        
        self.documents = all_documents
        self.chunks = [doc.page_content for doc in all_documents]
        
        self.save_vector_store()
        
        logger.info(f"Created embeddings for {len(all_documents)} chunks from {len(files)} files")
    
    def add_document(self, file_path: str):
        """
        Add a single document to the existing vector store.
        
        Args:
            file_path: Path to the document to add
        """
        logger.info(f"Adding document: {file_path}")
        
        documents = self._process_document(file_path)
        if not documents:
            return
        
        if self.vector_store is None:
            self.vector_store = InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            self.documents = documents
            self.chunks = [doc.page_content for doc in documents]
        else:
            self.vector_store.add_documents(documents)
            self.documents.extend(documents)
            self.chunks.extend([doc.page_content for doc in documents])
        
        logger.info(f"Added {len(documents)} chunks from {file_path}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as list of floats
        """
        embedding = self.embeddings.embed_query(query)
        return embedding
    
    def search_similar_documents(self, 
                                query: str, 
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with metadata
        """
        if self.vector_store is None:
            logger.error("No vector store loaded")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def save_vector_store(self):
        """Save the vector store and metadata to disk."""
        try:
            metadata = {
                'documents': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in self.documents
                ],
                'chunks': self.chunks,
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            summary_path = os.path.join(self.vector_store_path, "summary.json")
            summary = {
                'total_documents': len(set(doc.metadata.get('file_path', '') for doc in self.documents)),
                'total_chunks': len(self.documents),
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'files_processed': list(set(doc.metadata.get('file_path', '') for doc in self.documents)),
                'created_at': datetime.now().isoformat()
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Vector store metadata saved to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def load_vector_store(self):
        """Load the vector store and metadata from disk."""
        try:
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.documents = []
                for doc_data in metadata.get('documents', []):
                    doc = Document(
                        page_content=doc_data['page_content'],
                        metadata=doc_data['metadata']
                    )
                    self.documents.append(doc)
                
                self.chunks = metadata.get('chunks', [])
                
                if self.documents:
                    self.vector_store = InMemoryVectorStore.from_documents(
                        documents=self.documents,
                        embedding=self.embeddings
                    )
                
                if metadata.get('embedding_model') != self.embedding_model:
                    logger.warning(f"Model mismatch: stored={metadata.get('embedding_model')}, current={self.embedding_model}")
            
            logger.info(f"Vector store loaded from {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.vector_store is None:
            return {"status": "No vector store loaded"}
        
        unique_files = set(doc.metadata.get('file_path', '') for doc in self.documents)
        
        return {
            "total_documents": len(unique_files),
            "total_chunks": len(self.documents),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files": list(unique_files),
            "vector_store_initialized": self.vector_store is not None
        }
    
    def delete_document(self, file_path: str):
        """
        Remove a document from the vector store.
        Note: This requires rebuilding the index.
        
        Args:
            file_path: Path of the document to remove
        """
        old_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.metadata.get('file_path') != file_path]
        self.chunks = [doc.page_content for doc in self.documents]
        
        removed_count = old_count - len(self.documents)
        
        if removed_count > 0:
            if self.documents:
                self.vector_store = InMemoryVectorStore.from_documents(
                    documents=self.documents,
                    embedding=self.embeddings
                )
            else:
                self.vector_store = None
            
            logger.info(f"Removed {removed_count} chunks from {file_path}")
        else:
            logger.info(f"No chunks found for {file_path}")


def main():
    """Example usage of the EmbeddingsManager."""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    embeddings_manager = EmbeddingsManager(
        google_api_key=GOOGLE_API_KEY,
        embedding_model="models/embedding-001",
        knowledge_base_dir="knowledge_base",
        vector_store_path="vector_store"
    )
    
    embeddings_manager.create_embeddings_from_directory()
    
    stats = embeddings_manager.get_stats()
    print("Vector Store Statistics:")
    print(json.dumps(stats, indent=2))
    
    query = "What is machine learning?"
    results = embeddings_manager.search_similar_documents(query, top_k=3)
    
    print(f"\nTop 3 results for query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. File: {result['metadata'].get('file_path', 'Unknown')}")
        print(f"   Similarity: {result['similarity_score']:.4f}")
        print(f"   Content: {result['content'][:100]}...")
        print()


if __name__ == "__main__":
    main()

# Create sample RAG implementation code
rag_implementation_code = '''
"""
RAG System Implementation for Growth Intelligence
Core components for document retrieval and generation
"""

import os
from typing import List, Dict, Any
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import numpy as np

class RAGPipeline:
    """
    End-to-end RAG pipeline for business document search and generation
    """
    
    def __init__(self, openai_api_key: str, persist_directory: str = "./chroma_db"):
        """Initialize RAG pipeline with embeddings and vector store"""
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template("""
        You are a business intelligence assistant. Use the following context to answer the question.
        Be precise and cite sources. If you don't know, say so.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer with sources:
        """)
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load and process documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            # Read file content (simplified - add PDF, DOCX loaders as needed)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "filename": os.path.basename(file_path)
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_vectorstore(self, documents: List[Document]):
        """Create and persist vector store from documents"""
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist the database
        self.vectorstore.persist()
        print(f"Created vector store with {len(splits)} document chunks")
    
    def load_vectorstore(self):
        """Load existing vector store"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        
        # Similarity search with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "confidence": max(0, 1 - score)  # Convert distance to confidence
            })
        
        return results
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        """Generate answer using retrieved context"""
        # Format context from retrieved documents
        context = "\\n\\n".join([
            f"Source: {doc['metadata']['filename']}\\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        # Generate response
        response = self.llm.invoke(
            self.rag_prompt.format(context=context, question=query)
        )
        
        return {
            "answer": response.content,
            "sources": [doc['metadata']['filename'] for doc in retrieved_docs],
            "confidence": np.mean([doc['confidence'] for doc in retrieved_docs]),
            "context_used": len(retrieved_docs)
        }
    
    def search_and_answer(self, query: str, k: int = 5) -> Dict:
        """End-to-end search and answer generation"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, k=k)
        
        # Generate answer
        answer_result = self.generate_answer(query, retrieved_docs)
        
        # Combine results
        return {
            "query": query,
            "answer": answer_result["answer"],
            "sources": answer_result["sources"],
            "confidence": answer_result["confidence"],
            "retrieved_documents": retrieved_docs,
            "metrics": {
                "documents_retrieved": len(retrieved_docs),
                "avg_similarity": np.mean([doc['similarity_score'] for doc in retrieved_docs]),
                "confidence_score": answer_result["confidence"]
            }
        }

# Example usage and evaluation
def evaluate_rag_system(rag_pipeline: RAGPipeline, test_queries: List[str]) -> Dict:
    """Evaluate RAG system performance"""
    results = []
    
    for query in test_queries:
        result = rag_pipeline.search_and_answer(query)
        results.append(result)
    
    # Calculate metrics
    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_docs_retrieved = np.mean([r['metrics']['documents_retrieved'] for r in results])
    avg_similarity = np.mean([r['metrics']['avg_similarity'] for r in results])
    
    return {
        "evaluation_metrics": {
            "average_confidence": avg_confidence,
            "average_documents_retrieved": avg_docs_retrieved,
            "average_similarity_score": avg_similarity,
            "total_queries_processed": len(test_queries)
        },
        "individual_results": results
    }

# Sample data for testing
sample_documents = [
    {
        "filename": "marketing_campaign_q3.txt",
        "content": "Q3 Marketing Campaign Results: Email personalization increased conversion rates by 15% (treatment: 27%, control: 12%). High CLV customers showed 22% uplift. Budget: $50K, Revenue generated: $125K, ROI: 2.5x."
    },
    {
        "filename": "product_launch_analysis.txt", 
        "content": "Product Launch Analysis: New premium features drove 23% engagement increase among early adopters. Premium tier adoption rate: 2.3x higher than standard tier. Churn reduced by 8% in first month post-launch."
    },
    {
        "filename": "customer_retention_study.txt",
        "content": "Customer Retention Study: Loyalty program implementation resulted in 18% overall churn reduction. Segment analysis: Millennials (-25% churn), Gen X (-12% churn), Boomers (-15% churn). Program cost: $30K, Retention value: $200K."
    }
]

# Test queries for evaluation
test_queries = [
    "What was the ROI of the Q3 marketing campaign?",
    "How did the premium features affect customer engagement?", 
    "Which customer segments benefited most from the loyalty program?",
    "What was the total budget spent on retention initiatives?",
    "How much revenue was generated from email personalization?"
]

if __name__ == "__main__":
    # Example implementation
    print("RAG Pipeline Implementation Example")
    print("=" * 50)
    
    # Note: In real implementation, you would:
    # 1. Set up OpenAI API key
    # 2. Create actual document files
    # 3. Initialize and use the RAG pipeline
    
    print("\\nSample RAG Pipeline Components:")
    print("✓ Document ingestion and chunking")
    print("✓ Vector embeddings and similarity search") 
    print("✓ LLM-powered answer generation")
    print("✓ Source attribution and confidence scoring")
    print("✓ Evaluation metrics calculation")
    
    print("\\nNext steps:")
    print("1. Set up OpenAI API key in environment")
    print("2. Create sample business documents")
    print("3. Initialize RAG pipeline and test queries")
    print("4. Evaluate and optimize retrieval performance")
'''

# Save the code to a file
with open("rag_implementation.py", "w") as f:
    f.write(rag_implementation_code)
    
print("Created rag_implementation.py")
print("=" * 50)
print("Contents:")
print("- RAGPipeline class with full implementation")
print("- Document loading, chunking, and embedding")  
print("- Vector store creation and similarity search")
print("- LLM-powered answer generation with citations")
print("- Evaluation framework with metrics")
print("- Sample data and test queries")
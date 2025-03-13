import os
import chromadb
import numpy as np
import ollama
import torch
from sentence_transformers import SentenceTransformer

class ArabicRAGSystem:
    def __init__(self, 
                 text_file_path=None, 
                 db_path='./chroma_db', 
                 embedding_model='BAAI/bge-m3',  
                 ollama_model='qwen2'):
        """
        Initialize the RAG system with embedding-based retrieval and re-ranking.
        """
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "arabic_documents"

        # Load embedding model (same for retrieval & re-ranking)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Set LLM model
        self.ollama_model = ollama_model
        
        # Create collection if it doesn't exist
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)

        # Process documents if provided
        if text_file_path and self.collection.count() == 0:
            self.process_documents(text_file_path)

    def format_based_chunking(self, text):
        """
        Splits text into chunks based on blank lines.
        Each paragraph remains a single chunk.
        """
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        return chunks

    def process_documents(self, file_path):
        """
        Reads, formats, chunks, embeds, and stores documents in ChromaDB.
        """
        print("Processing documents...")

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Use format-based chunking for better retrieval
        chunks = self.format_based_chunking(text)

        # Embed and store chunks
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk).tolist()
            self.collection.add(
                embeddings=embedding,
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )

        print(f"Processed {len(chunks)} chunks and stored in database.")

    def retrieve_relevant_context(self, query, top_k=2):
        """
        Retrieve the most relevant context for a given query using ChromaDB.
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        # Retrieve initial results from ChromaDB (no re-ranking)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        # Ensure valid results
        if not results or 'documents' not in results or not results['documents']:
            print("⚠️ No relevant context found.")
            return []
        
        print("Retrived: ", results['documents'][0])

        return results['documents'][0]  # Directly return top-k documents
    
    def generate_response(self, query):
        """
        Generates a structured and precise response using retrieved and re-ranked context.
        """
        context = self.retrieve_relevant_context(query)

        # Structured Prompt
        prompt = f"""
    **🔹 تعليمات صارمة:**
    - مهمتك هي تقديم إجابة **دقيقة** تعتمد حصريًا على المعلومات التالية مع التركيز على السؤال:
    جاوب بناءً على السؤال المعطى.
    ** انت مساعد اكاديمي في جامعة حائل **
    ** الجامعة تتبع معدل الطالب من 4 **
    ** جاوب بناءً على جامعة حائل **

    🔹 **المعلومات المتاحة:**
    {''.join(context)}

    ❓ **السؤال:** {query}

    ✅ **التعليمات:**
    1️⃣ **إذا كان الجواب موجودًا في المعلومات المتاحة، استخدمه حرفيًا أو قم بإعادة صياغته لتحسين الوضوح.**
    2️⃣ **إذا لم يكن الجواب واضحًا، فقم باستخراج المعلومات ذات الصلة وقدمها بأفضل طريقة ممكنة.**
    3️⃣ **🚨 لا يجوز لك رفض الإجابة طالما هناك أي معلومة ذات صلة في النص المسترجع.**
    4️⃣ **🚨 لا يجوز لك إضافة معلومات جديدة أو الاستنتاج من خارج المعلومات المسترجعة.**
    5️⃣ **إذا لم تجد أي إجابة في المعلومات، عندها فقط يمكنك القول: "⚠️ لا توجد معلومات كافية."**
    """

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': 'أنت مساعد أكاديمي. أجب باللغة العربية فقط. تأكد من دقة الإجابة والتزم بالتنسيق المطلوب دون إضافة معلومات غير مؤكدة.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            print("The prompt is : ", prompt)
            return response['message']['content']
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "حدث خطأ أثناء توليد الاستجابة."

# ----------------------------
#  Database Initialization & Query Execution
# ----------------------------

def initialize_database(file_path):
    """
    Initialize ChromaDB with formatted document storage.
    """
    rag_system = ArabicRAGSystem(text_file_path=file_path)
    print("Database initialized successfully.")

def query_database(query):
    """
    Query the system and return an accurate response.
    """
    rag_system = ArabicRAGSystem()
    return rag_system.generate_response(query)

# ----------------------------
#  Example Usage
# ----------------------------

def main():
    # Step 1: Initialize database (run once)
    initialize_database('university_rules_last.txt')
    
    # Step 2: Query the database
    query = "ماهي شروط التحويل الداخلي؟" 
    response = query_database(query)
    print(response)

if __name__ == "__main__":
    main()

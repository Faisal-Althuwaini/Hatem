import os
import chromadb
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ArabicRAGSystem:
    def __init__(self, 
                 text_files=None,  # دعم عدة ملفات
                 db_path='./chroma_db', 
                 embedding_model='BAAI/bge-m3',  
                 ollama_model='qwen2'):
        """
        نظام RAG لمعالجة واسترجاع المعلومات من عدة ملفات نصية مع دعم إعادة الترتيب الذكي (Re-ranking).
        """
        os.makedirs(db_path, exist_ok=True)
        
        # تهيئة قاعدة البيانات
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "arabic_documents"

        # تحميل نموذج التضمين الذي يدعم العربية
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # تحديد نموذج Ollama
        self.ollama_model = ollama_model
        
        # إنشاء المجموعة في قاعدة البيانات إذا لم تكن موجودة
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)

        # معالجة الملفات إذا كانت موجودة ولم يتم تخزينها مسبقًا
        if text_files and self.collection.count() == 0:
            self.process_documents(text_files)

    def format_based_chunking(self, text):
        """ تقسيم النصوص إلى فقرات بناءً على الأسطر الفارغة """
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    def process_documents(self, file_paths):
        """ معالجة عدة ملفات نصية وإضافتها إلى قاعدة البيانات """
        print("📥 جاري معالجة الملفات...")

        for file_path in file_paths:
            file_name = os.path.basename(file_path).split('.')[0]  # استخراج اسم الملف بدون الامتداد
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            chunks = self.format_based_chunking(text)

            # تخزين كل جزء في قاعدة البيانات مع معرف فريد
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[f"{file_name}_chunk_{i}"]  # إضافة اسم الملف لمنع التكرار
                )

            print(f"✅ تمت معالجة {len(chunks)} جزء من ملف {file_name}")

    def retrieve_relevant_context(self, query, top_k=10):
        """ استرجاع أولي للنتائج من ChromaDB بدون ترتيب """
        processed_query = preprocess_query(query)  # Normalize the query

        query_embedding = self.embedding_model.encode(processed_query).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        retrieved_docs = results.get('documents', [[]])[0]

        return retrieved_docs if retrieved_docs else []

    def rerank_results(self, query, retrieved_docs):
        """ إعادة ترتيب النتائج بناءً على التشابه الدلالي باستخدام BAAI/bge-m3 """
        
        if not retrieved_docs:
            return []

        # تحويل الاستعلام إلى تضمين (Embedding)
        query_embedding = self.embedding_model.encode(query).reshape(1, -1)

        # تحويل الوثائق المسترجعة إلى تضمينات
        doc_embeddings = np.array([self.embedding_model.encode(doc) for doc in retrieved_docs])

        # حساب التشابه باستخدام Cosine Similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # ترتيب الوثائق بناءً على أعلى تشابه
        sorted_docs = [doc for _, doc in sorted(zip(similarities, retrieved_docs), reverse=True)]

        return sorted_docs

    def retrieve_and_rerank(self, query, top_k=10):
        """ استرجاع النتائج ثم إعادة ترتيبها للحصول على الإجابة الأكثر دقة """
        retrieved_docs = self.retrieve_relevant_context(query, top_k=top_k)
        reranked_docs = self.rerank_results(query, retrieved_docs)
        return reranked_docs

    def generate_response(self, query):
        """ توليد إجابة باستخدام الوثائق الأكثر صلة """
        retrieved_docs = self.retrieve_and_rerank(query, top_k=10)
        

        # استخدام أفضل 3 وثائق بعد إعادة الترتيب
        context = "\n\n".join(retrieved_docs[:3])

        print("Context: " ,context)
        prompt = f"""
        🔹 لا تخلط بين اللغات، لا تستخدم أي رموز أو أرقام بلغات أخرى.

        🔹 **المعلومات المتاحة:**  
        {context}

        ❓ **السؤال:** {query}

        ✅ **التعليمات الخاصة بالإجابة:**
        1️⃣ **إذا كان الجواب موجودًا في المعلومات المتاحة، استخدمه حرفيًا أو قم بإعادة صياغته لتحسين الوضوح.**
        2️⃣ **إذا لم يكن الجواب واضحًا، فقم باستخراج المعلومات ذات الصلة وقدمها بأفضل طريقة ممكنة.**
        3️⃣ **🚨 لا يجوز لك رفض الإجابة طالما هناك أي معلومة ذات صلة في النص المسترجع.**
        4️⃣ **🚨 لا يجوز لك إضافة معلومات جديدة أو الاستنتاج من خارج المعلومات المسترجعة.**
        ** تأكد من فهم السؤال والجواب بدقة على السؤال من المعلومات المتاحة **

        """

        system_prompt = f"""
🔹 **أنت مساعد أكاديمي في جامعة حائل.** 
🔹 دورك هو تقديم إجابات دقيقة بناءً على المعلومات المتاحة فقط.
🔹 لا يجوز لك استخدام أي معلومات من خارج النص المسترجع.
🔹 يجب أن تلتزم بالإجابة وفقًا لنظام جامعة حائل.
🔹 الجامعة تتبع نظام المعدل من 4، فلا تستخدم أي نظام آخر.
🔹 لا تستنتج معلومات غير موجودة في البيانات المتاحة.
🔹 إذا لم تجد أي إجابة في النصوص المتاحة، يجب أن تقول بوضوح: "⚠️ لا توجد معلومات كافية."
        """

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']

        except Exception as e:
            return f"⚠️ خطأ: {str(e)}"

# normalizing functions
import re

def remove_diacritics(text):
    """Remove Arabic diacritics (Tashkeel)."""
    arabic_diacritics = re.compile(r'[\u064B-\u065F]')
    return arabic_diacritics.sub('', text)

def normalize_arabic(text):
    """Normalize Arabic letters for consistency."""
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")  # Normalize Alef variations
    text = text.replace("ة", "ه")  # Convert Ta Marbuta to Ha
    return text

def preprocess_query(query):
    """Apply normalization steps to query."""
    query = remove_diacritics(query)  # Remove diacritics
    query = normalize_arabic(query)  # Normalize letters
    return query  # Keep punctuation and spaces as they are

# ----------------------------
#  تهيئة قاعدة البيانات واستعلام المستخدم
# ----------------------------


def initialize_database(file_paths):
    """ تهيئة قاعدة البيانات وإضافة عدة ملفات """
    rag_system = ArabicRAGSystem(text_files=file_paths)
    print("✅ تمت تهيئة قاعدة البيانات بنجاح.")

def query_database(query):
    """ استعلام النظام للحصول على إجابة دقيقة """
    rag_system = ArabicRAGSystem()
    return rag_system.generate_response(query)

# ----------------------------
#  تشغيل البرنامج
# ----------------------------

def main():
    # قائمة الملفات التي نريد إضافتها
    files = ['norm_studies_and_exams.txt', 'norm_student_rights_and_duties.txt', 'norm_student_guide.txt', 'norm_student_box.txt', 'norm_conducts_and_disc.txt']
    
    # خطوة 1: تهيئة قاعدة البيانات وإضافة الملفات
    initialize_database(files) 
    
    # خطوة 2: استعلام
    query = "ما هي تخصصات كلية الحاسب ؟" 

    response = query_database(query)
    print(response)

if __name__ == "__main__":
    main()

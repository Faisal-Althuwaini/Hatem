import os
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

class ArabicRAGSystem:
    def __init__(self, 
                 text_files=None,  # دعم عدة ملفات
                 db_path='./chroma_db', 
                 embedding_model='BAAI/bge-m3',  
                 ollama_model='qwen2'):
        """
        نظام RAG لمعالجة واسترجاع المعلومات من عدة ملفات نصية.
        """
        os.makedirs(db_path, exist_ok=True)
        
        # تهيئة قاعدة البيانات
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "arabic_documents"

        # تحميل نموذج التضمين
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

    def retrieve_relevant_context(self, query, top_k=5):
        """ استرجاع السياق الأكثر صلة للاستعلام """
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        retrieved_docs = results.get('documents', [[]])[0]
            # تأكد أن هناك نتائج
        if not retrieved_docs:
            return []
        
        # **فرز النتائج بناءً على التشابه**
        ranked_docs = sorted(retrieved_docs, key=lambda doc: self.embedding_model.encode(doc).dot(query_embedding), reverse=True)
        
        return ranked_docs


    def generate_response(self, query):
        """ توليد إجابة بناءً على السياق المسترجع """
        context = self.retrieve_relevant_context(query)

        prompt = f"""
        **🔹 تعليمات صارمة:**
        - مهمتك هي تقديم إجابة **دقيقة** تعتمد حصريًا على المعلومات التالية مع التركيز على السؤال:
        - أنت مساعد أكاديمي في جامعة حائل، جاوب فقط بالمعلومات المتاحة.
        - الجامعة تتبع معدل الطالب من 4
        - لا يجوز لك إضافة معلومات جديدة أو الاستنتاج من خارج المعلومات المسترجعة.
        - إذا لم يكن الجواب واضحًا، فقم باستخراج المعلومات ذات الصلة وقدمها بأفضل طريقة ممكنة
        - 🚨 **لا تقل "لا توجد معلومات" إلا إذا لم يكن هناك أي نص متاح للاسترجاع.**
        - 🚨 **لا تضف معلومات جديدة من خارج السياق.**

        
        🔹 **المعلومات المتاحة:**
        {''.join(context)}

        ❓ **السؤال:** {query}
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': 'أنت مساعد أكاديمي. أجب باللغة العربية فقط بدقة دون معلومات إضافية.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']
        
        except Exception as e:
            return f"⚠️ خطأ: {str(e)}"

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
    # files = ['studies_and_exams.txt', 'student_rights_and_duties.txt', 'student_guide.txt', 'student_box.txt', 'conducts_and_disc.txt']
    
    # خطوة 1: تهيئة قاعدة البيانات وإضافة الملفات
    # initialize_database(files) 
    
    # خطوة 2: استعلام
    query = "ما هي تخصصات كلية الادارة؟" 
    response = query_database(query)
    print(response)

if __name__ == "__main__":
    main()

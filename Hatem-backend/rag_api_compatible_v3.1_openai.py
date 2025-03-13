import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import openai
from dotenv import load_dotenv

load_dotenv()


class ArabicRAGSystem:
    def __init__(self, 
                 text_files=None,  
                 db_path='./chroma_db', 
                 embedding_model='BAAI/bge-m3',  
                 openai_model='gpt-3.5-turbo'):
        """
        نظام RAG لمعالجة واسترجاع المعلومات من عدة ملفات نصية مع دعم إعادة الترتيب الذكي (Re-ranking).
        """
        os.makedirs(db_path, exist_ok=True)
        
        # تهيئة قاعدة البيانات
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "arabic_documents"

        # تحميل نموذج التضمين الذي يدعم العربية
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # تحديد نموذج OpenAI
        self.openai_model = openai_model
        self.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set as an environment variable
        
        if not self.api_key:
            raise ValueError("⚠️ لم يتم العثور على مفتاح OpenAI API! الرجاء ضبط 'OPENAI_API_KEY' في متغيرات البيئة.")

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
            file_name = os.path.basename(file_path).split('.')[0]  
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            chunks = self.format_based_chunking(text)

            # تخزين كل جزء في قاعدة البيانات مع معرف فريد
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[f"{file_name}_chunk_{i}"]
                )

            print(f"✅ تمت معالجة {len(chunks)} جزء من ملف {file_name}")

    def retrieve_relevant_context(self, query, top_k=5):
        """ استرجاع أولي للنتائج من ChromaDB """
        processed_query = preprocess_query(query)
        query_embedding = self.embedding_model.encode(processed_query).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        retrieved_docs = results.get('documents', [[]])[0]

        return retrieved_docs if retrieved_docs else []

    def rerank_results(self, query, retrieved_docs):
        """ إعادة ترتيب النتائج بناءً على التشابه الدلالي """
        if not retrieved_docs:
            return []

        query_embedding = self.embedding_model.encode(query).reshape(1, -1)
        doc_embeddings = np.array([self.embedding_model.encode(doc) for doc in retrieved_docs])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        sorted_docs = [doc for _, doc in sorted(zip(similarities, retrieved_docs), reverse=True)]
        return sorted_docs

    def retrieve_and_rerank(self, query, top_k=5):
        """ استرجاع النتائج ثم إعادة ترتيبها للحصول على الإجابة الأكثر دقة """
        retrieved_docs = self.retrieve_relevant_context(query, top_k=top_k)
        return self.rerank_results(query, retrieved_docs)

    def generate_response(self, query):
        """ توليد إجابة باستخدام الوثائق الأكثر صلة """
        retrieved_docs = self.retrieve_and_rerank(query, top_k=5)

        context = "\n\n".join(retrieved_docs[:3])
        print("Context: ", context)

        prompt = f"""
        🔹 لا تخلط بين اللغات، لا تستخدم أي رموز أو أرقام بلغات أخرى.
        🔹 جاوب على السؤال مباشرة **بدون أي مقدمات أو عبارات تفسيرية**.
        🔹 لا تبدأ الجواب بكلمات مثل "الإجابة:"، "نعم، في المعلومات المتاحة..."، "بناءً على البيانات المتاحة..."، بل **أجب مباشرة**.

        🔹 المعلومات المتاحة:   
        {context}

        ❓ السؤال: {query}

        ✅ التعليمات الخاصة بالإجابة:
        1️⃣ استخدم المعلومات المتاحة حرفيًا أو أعد صياغتها بوضوح.
        2️⃣ لا تبدأ الجواب بأي كلمة مثل "الإجابة:" أو "الرد:".
        3️⃣ 🚨 لا يجوز لك رفض الإجابة طالما هناك أي معلومة ذات صلة.
        4️⃣ 🚨 لا تضف معلومات جديدة أو تستنتج من خارج النصوص المتاحة.

        🎯 **مثال صحيح:**
        ❌ "نعم، في المعلومات المتاحة يوجد تخصص قانون..."
        ✅ "تخصص القانون موجود ضمن كلية الشريعة والقانون."
        """

        system_prompt = f"""
🔹 أنت مساعد أكاديمي في جامعة حائل.
🔹 دورك هو تقديم إجابات دقيقة بناءً على المعلومات المتاحة فقط.
🔹 🚨 لا تبدأ الجواب بأي كلمة مثل "الإجابة:" أو "الرد:" أو "وفقًا للمعلومات المتاحة"، بل قدمه مباشرة.
🔹 لا يجوز لك استخدام أي معلومات من خارج النص المسترجع.
🔹 يجب أن تلتزم بالإجابة وفقًا لنظام جامعة حائل.
🔹 الجامعة تتبع نظام المعدل من 4، فلا تستخدم أي نظام آخر.
🔹 لا تستنتج معلومات غير موجودة في البيانات المتاحة.
🔹 إذا لم تجد أي إجابة في النصوص المتاحة، يجب أن تقول بوضوح: "⚠️ لا توجد معلومات كافية."
"""
        
        try:
            client = openai.OpenAI(api_key=self.api_key)  

            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ خطأ: {str(e)}"

# ----------------------------
#  دوال معالجة النصوص
# ----------------------------

def remove_diacritics(text):
    """ إزالة التشكيل """
    arabic_diacritics = re.compile(r'[\u064B-\u065F]')
    return arabic_diacritics.sub('', text)

def normalize_arabic(text):
    """ توحيد شكل الحروف العربية """
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    return text

def preprocess_query(query):
    """ معالجة الاستعلامات """
    query = remove_diacritics(query)
    query = normalize_arabic(query)
    return query

# ----------------------------
#  تشغيل البرنامج
# ----------------------------

def main():
    files = ['norm_studies_and_exams.txt', 'norm_student_rights_and_duties.txt']
    rag_system = ArabicRAGSystem(text_files=files) 
    query = "ما هي تخصصات كلية الحاسب؟"
    response = rag_system.generate_response(query)
    print(response)

if __name__ == "__main__":
    main()

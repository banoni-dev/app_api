from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
app = Flask(__name__)

   

@app.route('/search', methods=['GET'])
def search():
    query = request.args['question']
    
    if not query:
        return jsonify({'message': 'Question is required'}), 400

    # Charger le modèle de embeddings multilingue
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # Exemple de textes en arabe
    texts = [
       "الوثائق مطلوبة لإستبدال بطاقة تعريف الوطنية هي تعمير المطبوعة الإدارية وإمضاؤهامضمون ولادة لم يمض على تاريخ إصداره أكثر من ثلاثة أشهر ثلاثة صور فوتوغرافية للمعني بالأمر تكون من حجم 4/3 صم وتؤخذ وجها على لوحة خلفية من اللون الأبيض أو الفاتح وبمقياس 10/1 وتبين الشعر والعينين نسخة مصورة من البطاقة المطلوب تعويضها مع الاستظهار بالأصل الذي يحتفظ به المعني على أن يسلمه للمصالح المختصة عند تسلم البطاقة الجديدة وثيقة مثبتة لتغيير عناصر الحالة المدنية أو المقر أو المهنة (عند الاقتضاء) شهادة حضور بالنسبة للطلبة لم يمضي على تاريخ إستلامها ثلاثة أشهر وصل خلاص تسلمه قباضات المالية",
       "الوثائق المطلوبة لإستخراج بطاقة تعريف وطنية لأول مرة :مطبوعة إدارية للتعمير.مضمون ولادة مستخرج من السجلات الأصلية للحالة المدنية لم يمض على تاريخ إصداره أكثر من ثلاثة أشهر شهادة في الجنسية التونسية- شهادة إقامة شهادة عمل أو شهادة حضور مدرسي أو جامعي ثلاثة صور فوتوغرافية للمعني بالأمر تكون من حجم 4/3 صم وتؤخذ وجها على لوحة خلفية من اللون الأبيض أو الفاتح وبمقياس 10/1 وتبين الشعر والعينين وصل خلاص تسلمه قباضات المالية قيمته 3 دنانير نسخة مصورة من بطاقة إقامة ونسخة من بطاقة القنصلية بالنسبة للمقيمين بالخارج(في نظيرين) ترخيص من الولي الشرعي معرف بالإمضاء ومعلل في الغرض لإستخراج البطاقة : (مهني، تربوي، رياضي أو بدني) بالنسبة للقصر دون 18 سنة شهادة في الفصيلة الدموية (اختيارية)",
       "الوثائق المطلوبة للإستخراج جواز سفر هي: تعمير إستمارة الحصول على جواز سفر عادي مقروء آليا وإمضاؤها بصفة شخصية داخل الخانة المعدة للغرض نسخة من بطاقة التعريف الوطنية مع الاستظهار بالأصل أو مضمون ولادة بالنسبة للقصر04 صور شمسية حسب المواصفات التالية: أن تكون خلفية الصورة بيضاء أن يكون حجم الصورة 3.5/4.5 صم تقريبا.ما يفيد الدراسة بالنسبةللتلاميذ والطلبة.ترخيص الولي بالنسبة للقصر مصحوبا بنسخة من بطاقة تعريف و وصل خلاص تسلمه قباضات المالية قيمته:25 دنانيرا بالنسبة للطلبة والتلاميذ الذين أثبتوا صفتهم تلك بتقديم شهادة أوالأطفال الذين لم يبلغوا سنّ السابعة وكذلك التمديد في صلوحيتها،80 دينارا بالنسبة للبقية وكذلك التمديد في صلوحيتها،إضافة الجواز القديم في حالة تجديد،تقديم طلب على ورق عادي عند الرغبة في الإحتفاظ بالجواز القديم",
       "فتح حساب إدخار :مطبوعة فتح حساب متوفرة في البريد،نسخة من بطاقة تعريف،المبلغ المالي الذي سيتم إدخاره",
       "فتح حساب جاري:شهادة عمل أصلية وليست نسخة،نسخة من بطاقة تعريف الوطنية،مطبوعة فتح حساب جاري ومطبوعة مصاحبه لها (متوفر في البريد)،10د معلوم الخدمة"
    ]

    # Générer les embeddings pour les textes
    embeddings = model.encode(texts)
    # Convertir les embeddings en une matrice NumPy
    embeddings = np.array(embeddings)
    # Utiliser FAISS pour créer une instance de recherche de vecteurs
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Dimension des embeddings
    index.add(embeddings)
    # Fonction pour rechercher un document similaire
    def search_document(query, k=1):
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k)  # k pour trouver les k voisins les plus proches
        return [(texts[i], D[0][j]) for j, i in enumerate(I[0])] 
    
    results = search_document(query, k=1)  # Trouver le document le plus proche
    
    if results:
        best_result = results[0]
        return jsonify({'repense': best_result[0]}), 200
    else:
        return jsonify({'message': 'لم يتم العثور على إجابة.'}), 404

if __name__ == '__main__':
    app.run(host='192.168.1.118', port=10001)
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# تحميل النموذج مع معالجة الأخطاء
@st.cache_resource
def load_my_model():
    try:
        model = load_model("weather_model.h5")
        st.success("✅ تم تحميل النموذج بنجاح")
        return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
        return None

model = load_my_model()

# واجهة المستخدم
st.set_page_config(page_title="🌦️ مصنف الطقس", layout="centered")
st.title("🌈 تصنيف أحوال الطقس من الصور")
st.markdown("---")

# قسم تحميل الصورة
st.header("📤 تحميل صورة")
uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # معالجة الصورة
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="الصورة المدخلة", width=300)
        
        # تحويل الصورة للنموذج
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التنبؤ
        st.write("🔮 جاري تحليل الصورة...")
        prediction = model.predict(img_array)
        class_names = ["Cloudy", "Rain", "Shine", "Sunrise"]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # عرض النتائج
        st.success(f"✅ النتيجة: {predicted_class}")
        st.metric("مستوى الثقة", f"{confidence:.2%}")
        
        # رسم بياني
        st.subheader("توزيع الاحتمالات:")
        prob_data = {"الفئة": class_names, "الاحتمال": prediction[0]}
        st.bar_chart(prob_data, x="الفئة", y="الاحتمال")
        
    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")

# قسم معلومات إضافية
st.markdown("---")
st.info("""
### تعليمات الاستخدام:
1. قم بتحميل صورة لحالة جوية
2. انتظر حتى يكمل النموذج التحليل
3. شاهد النتائج والتوقعات
""")

# التحقق من وجود الملفات
st.sidebar.header("فحص النظام")
st.sidebar.write("الملفات الموجودة:")
for file in os.listdir():
    st.sidebar.write(f"- {file}")
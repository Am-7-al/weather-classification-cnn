import os
import splitfolders
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# 1. التحقق من المجلد وتقسيم البيانات
# ==============================
input_folder = r"C:\Users\am\Downloads\wather\archive\Multi-class Weather Dataset"
output_folder = r"C:\Users\am\Downloads\wather\weather_split"

# إنشاء المجلد إذا لم يكن موجوداً
os.makedirs(output_folder, exist_ok=True)

if not os.listdir(output_folder):  # إذا كان المجلد فارغاً
    print("🔄 جاري تقسيم البيانات (70% Train, 20% Val, 10% Test)...")
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .2, .1))
else:
    print("✅ تم العثور على بيانات مقسمة مسبقاً.")

# ==============================
# 2. تحميل البيانات
# ==============================
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")
test_dir = os.path.join(output_folder, "test")

# التحقق من وجود المجلدات
for folder in [train_dir, val_dir, test_dir]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"المجلد {folder} غير موجود!")

# إنشاء مولدات البيانات
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical')

val_gen = val_datagen.flow_from_directory(
    val_dir, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical')

test_gen = test_datagen.flow_from_directory(
    test_dir, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical', 
    shuffle=False)  # مهم للتحليل لاحقاً

num_classes = len(train_gen.class_indices)
print(f"🔢 عدد الفئات: {num_classes} → {list(train_gen.class_indices.keys())}")

# ==============================
# 3. بناء النموذج CNN
# ==============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

# ==============================
# 4. تدريب النموذج
# ==============================
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=10,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size
)

# ==============================
# 5. تقييم النموذج
# ==============================
test_loss, test_acc = model.evaluate(test_gen)
print(f"🎯 دقة النموذج على بيانات الاختبار: {test_acc:.2%}")

# حفظ النموذج
model.save("weather_model.h5")
print("✅ تم حفظ النموذج: weather_model.h5")

# ==============================
# 6. مصفوفة الالتباس والتقرير
# ==============================
# الحصول على التوقعات
y_true = test_gen.classes
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

# مصفوفة الالتباس
cm = confusion_matrix(y_true, y_pred_classes)
labels = list(test_gen.class_indices.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=labels, yticklabels=labels)
plt.xlabel("التوقعات")
plt.ylabel("القيم الحقيقية")
plt.title("مصفوفة الالتباس")
plt.show()

# تقرير التصنيف
print("\nتقرير التصنيف:\n")
print(classification_report(y_true, y_pred_classes, target_names=labels))

# رسم دقة التدريب والتحقق
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='دقة التدريب')
plt.plot(history.history['val_accuracy'], label='دقة التحقق')
plt.title('دقة النموذج')
plt.ylabel('الدقة')
plt.xlabel('العصر')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='خسارة التدريب')
plt.plot(history.history['val_loss'], label='خسارة التحقق')
plt.title('خسارة النموذج')
plt.ylabel('الخسارة')
plt.xlabel('العصر')
plt.legend()

plt.tight_layout()
plt.show()
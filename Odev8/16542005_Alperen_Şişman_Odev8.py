import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

# pip install pandas
# pip install numpy
# pip install scikit-learn
# pip install keras
# pip install tensorflow

veriler= pd.read_csv("csv/telefon_fiyat_degisimi.csv")

#Sınıf Sayısını Belirleme
label_encoder=LabelEncoder().fit(veriler.price_range)
labels=label_encoder.transform(veriler.price_range)
classes = list(label_encoder.classes_)
x = veriler.drop(["price_range"],axis=1);
y = labels

# Verilerin Standartlaştırılması
sc = StandardScaler()
x = sc.fit_transform(x);

# Eğitim ve test verilerinin hazırlanması
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2)

# çıktı değerlerinin kategorileştirilmesi
Y_train = to_categorical(Y_train)
Y_test  = to_categorical(Y_test)

# YSA modelinin oluşturulması
model = Sequential()
model.add(Dense(20,input_dim=20,activation="relu")) # Girdi katmanı 20 nöron
model.add(Dense(16,activation="relu"))# Ara katmanı 16 nöron
model.add(Dense(12,activation="relu"))# Ara katmanı 12 nöron
model.add(Dense(8,activation="relu"))# Ara katmanı 8 nöron
model.add(Dense(4,activation="softmax"))# Çıktı katmanı 4 nöron
model.summary()

# YSA modelinin derlenmesi
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# YSA modelinin eğitilmesi
model.fit(X_train,Y_train, validation_data=(X_test,Y_test), epochs=100)

# Gerekli değerlerin gösterilmesi
print("Ortalama Eğitim Kaybı: ",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim Başarımı: ",np.mean(model.history.history["accuracy"]))
print("Ortalama Doğrulama Kaybı: ",np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama Başarımı: ",np.mean(model.history.history["val_accuracy"]))


# Eğitimde doğrulama başarımlarının gösterilmesi
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epok")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show()


# Eğitimde doğrulama kayıplarının gösterilmesi
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epok")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show()
    
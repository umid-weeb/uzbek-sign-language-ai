import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("1. Ma'lumotlar bazasi (Dataset) o'qilmoqda...")
    # CSV faylni pandas orqali ochamiz
    df = pd.read_csv('data/dataset.csv')
    
    # Harflar (Natijalar) alohida, koordinatalar (X, Y) alohida ajratib olinadi
    X = df.drop('label', axis=1) # Koordinatalar (X)
    y = df['label']              # Harflar (Y)

    print("2. Ma'lumotlar o'rganish va test qilish uchun 2 qismga bo'linmoqda...")
    # Baza 80% o'qish uchun, 20% imtihon qilib tekshirish uchun bo'linadi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Sun'iy intellekt (Random Forest) o'qitilmoqda (Training)...")
    # Neyron tarmoqni yaratamiz va o'qishni boshlaymiz
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("4. O'qitilgan modelni test (imtihon) qilib ko'ramiz...")
    # Model ko'rmagan 20% ma'lumot orqali undan javob so'raymiz
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Modelning aniqligi (Accuracy): {accuracy * 100:.2f}%")

    print("5. O'rgatilgan MIYA (Model) kompyuterga saqlanmoqda...")
    # Modelni keyinchalik kamerada ishlatish uchun bitta fayl (pickle) qilib saqlaymiz
    with open('data/sign_language_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("🎉 Barcha ishlar muvaffaqiyatli yakunlandi! Model 'data/sign_language_model.pkl' nomli faylga saqlandi.")

if __name__ == "__main__":
    main()
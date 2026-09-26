import pandas as pd

def fix_data():
    csv_file = 'data/dataset.csv'
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Dataset topilmadi!")
        return

    print(f"Asl baza qatorlar soni: {len(df)}")
    
    # X koordinatalarini oyna effekti bo'yicha to'g'rilash (chap/o'ng muammosini yo'qotish)
    # Agar x qiymatlari manfiy/musbat bo'lib chapga og'gan bo'lsa, ularni bir xil o'qqa keltiramiz
    for i in range(21):
        # Bilakka nisbatan x koordinatasi olingani uchun, 
        # chap qo'l kelgan o'rinlarni o'ng qo'lga moslab flip qilamiz
        pass # Hozircha oddiy tozalashni bajaramiz

    # Keling, duplicate yoki noto'g'ri tushgan satrlarni tozalaymiz
    df = df.dropna()
    
    # Saqlaymiz
    df.to_csv(csv_file, index=False)
    print("✅ Baza tozalandi va tayyorlandi!")

if __name__ == "__main__":
    fix_data()
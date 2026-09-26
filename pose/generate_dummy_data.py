import csv
import random
import os

def main():
    # Siz taqdim etgan rasmdagi O'zbek lotin daktil alifbosi harflari
    # (Kirilldagi keraksiz harflar chiqarilib, O', G', SH, CH, NG qo'shilgan)
    letters = [
        'A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 
        'O_star', 'G_star', 'SH', 'CH', 'NG'
    ]

    os.makedirs('data', exist_ok=True)
    csv_file = 'data/dataset.csv'

    # Sarlavhalarni yaratish (Harf va 21 ta barmoq nuqtasining x,y koordinatalari)
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}'])

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Har bir harf uchun sun'iy ravishda 150 ta qator (kadr) ma'lumot yaratamiz
        for letter in letters:
            for _ in range(150):
                row = [letter]
                # 21 ta nuqta uchun tasodifiy (lekin normallashtirilgan) koordinatalar
                for _ in range(21):
                    x = round(random.uniform(-100, 100), 4)
                    y = round(random.uniform(-100, 100), 4)
                    row.extend([str(x), str(y)])
                    # row.extend([x, y])
                writer.writerow(row)

    print(f"✅ Bajarildi! Jami {len(letters)} ta harf uchun sun'iy ma'lumotlar bazasi generatsiya qilindi.")
    print(f"Baza jami {len(letters) * 150} qatordan iborat va '{csv_file}' fayliga saqlandi.")

if __name__ == "__main__":
    main()
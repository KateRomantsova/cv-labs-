# cv-labs

Лабораторні роботи з курсу комп'ютерного зору (Python, OpenCV).

## Структура

```
cv-labs/
├── README.md
├── requirements.txt
├── .gitignore
├── lab01/
│   ├── src/       код
│   ├── data/      малі вхідні дані або посилання на великі
│   ├── results/   числові результати (CSV)
│   ├── figures/   рисунки для звіту
│   └── report/    звіт у PDF
└── lab02/
```

## Встановлення

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## lab01. Середовище, подання зображень, колірні простори

Код кроків 1-14 лежить у `lab01/src/` (`step2.py` ... `step14.py`, `lab1.py` запускає частини A, B, C).
Скрипти читають файли за іменем з поточної папки, тому запускайте їх з папки з вхідними даними:

```
cd lab01\data
py ..\src\step9.py
```

Результати (CSV) зберігайте в `lab01/results/`, рисунки для звіту в `lab01/figures/`, звіт у PDF в `lab01/report/`.

## Дані

Зображення (`*.jpg`, `*.png`) у репозиторій не потрапляють (див. `.gitignore`). Для lab01 потрібні файли:
`photo.jpg`, `day.jpg`, `lamp.jpg`, `day_mask.png`, `lamp_mask.png`, `texture.jpg`.

Посилання на дані: (додайте посилання на хмарну папку)

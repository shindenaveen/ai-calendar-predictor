# AI Calendar Predictor Utility 🗓️🤖

A Python GUI tool that predicts annual cycle calendar data using machine learning. It supports both database and file-based input and exports predictions and SQL insert statements.

## 🔧 Features
- Predicts `APP_TYPE`, `CYCLE_TYPE`, and `SPECIAL_RUN_CD` for a future year
- Uses Random Forest Classifier on historical cycle calendar data
- Generates Excel output and SQL `INSERT` statements
- Fetches data from MSSQL DB or allows manual file input
- User-friendly GUI built with Tkinter
- Logs execution details with timestamped log files

## 🚀 Tech Stack
- **Language:** Python
- **ML Library:** scikit-learn
- **GUI:** Tkinter
- **Database:** MSSQL via pyodbc
- **Others:** Pandas, NumPy, OpenPyXL, Logging

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/shindenaveen/ai-calendar-predictor.git
   cd ai-calendar-predictor

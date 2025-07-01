import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pyodbc  # Import the pyodbc library for database connection
import threading
import logging

# Database connection details (replace with your actual credentials)
DATABASE_SERVER = '10.191.254.103'
DATABASE_NAME = 'CACM1AD4'
DATABASE_USER = 'PSMAGOPS'
DATABASE_PASSWORD = 'ABC_123' # database password.
DATABASE_TABLE = 'CYCLE_CALENDER'  # Table name in the database
OUTPUT_DIRECTORY = "/apps/CGIADV/gcstools/GCSAutomation/CycleCalenderUtility/Inputs"  # Directory to save the CSV file from the database
LOG_DIRECTORY = "/apps/CGIADV/gcstools/GCSAutomation/Logs"

# Configure logging
def setup_logging():
    """Sets up logging to a file with timestamp and script name."""
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIRECTORY, f"{script_name}_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"Logging started for {script_name} at {timestamp}")

setup_logging()

def generate_future_dates(start_date, end_date):
    """Generates a list of dates between a start and end date."""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def predict_future_data_ml(df, future_year):
    """
    Predicts future data using a machine learning model (Random Forest).

    Args:
        df: Pandas DataFrame containing the historical data.
        future_year: The year for which to generate predictions.

    Returns:
        Pandas DataFrame with predicted data for the future year.
    """
    logging.info(f"Starting prediction for future year: {future_year}")
    try:
        # 1. Data Preprocessing and Feature Engineering
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df['CALENDAR_DATE'] = pd.to_datetime(df['CALENDAR_DATE'], format='%d-%b-%y', errors='coerce')
        df = df.dropna(subset=['CALENDAR_DATE'])
        df.loc[:, 'DAY_OF_WEEK'] = df['CALENDAR_DATE'].dt.day_name()
        df.loc[:, 'YEAR'] = df['CALENDAR_DATE'].dt.year
        df.loc[:, 'MONTH'] = df['CALENDAR_DATE'].dt.month
        df.loc[:, 'DAY'] = df['CALENDAR_DATE'].dt.day

        # 2. Feature Encoding (Label Encoding for Categorical Features)
        label_encoders = {}
        categorical_cols = ['APP_TYPE', 'CYCLE_TYPE', 'SPECIAL_RUN_CD', 'DAY_OF_WEEK']
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on all unique values, including potential future values
            all_values = pd.concat([df[col], pd.Series(['N/A', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100'])], ignore_index=True).unique()
            
            # Filter out NaN values before fitting
            all_values = [val for val in all_values if pd.notna(val)]
            le.fit(all_values)
            
            # Transform only non-NaN values
            df.loc[df[col].notna(), col] = le.transform(df.loc[df[col].notna(), col])
            label_encoders[col] = le

        # 3. Prepare Data for ML
        features = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK']
        targets = ['APP_TYPE', 'CYCLE_TYPE', 'SPECIAL_RUN_CD']

        X = df[features]
        y = df[targets]

        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='most_frequent')
        X = imputer.fit_transform(X)
        
        # Convert y to numeric type if it's not already
        for target in targets:
            y[target] = pd.to_numeric(y[target], errors='coerce')
        
        # Drop rows with NaN in y after conversion
        y = y.dropna()
        X = X[y.index]

        # 4. Train-Test Split (for evaluation, optional for prediction)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Model Training (Random Forest)
        models = {}
        for target in targets:
            model = RandomForestClassifier(random_state=42)
            # Reshape y_train[target] to be a 1D array
            model.fit(X_train, y_train[target])
            models[target] = model

        # 6. Model Evaluation (optional)
        for target in targets:
            y_pred = models[target].predict(X_test)
            accuracy = accuracy_score(y_test[target], y_pred)
            logging.info(f"Accuracy for {target}: {accuracy}")

        # 7. Generate Future Dates
        start_date = datetime(future_year, 1, 1)
        end_date = datetime(future_year, 12, 31)
        future_dates = generate_future_dates(start_date, end_date)

        # 8. Create Future DataFrame
        future_df = pd.DataFrame({'CALENDAR_DATE': future_dates})
        future_df.loc[:, 'DAY_OF_WEEK'] = future_df['CALENDAR_DATE'].dt.day_name()
        future_df.loc[:, 'YEAR'] = future_df['CALENDAR_DATE'].dt.year
        future_df.loc[:, 'MONTH'] = future_df['CALENDAR_DATE'].dt.month
        future_df.loc[:, 'DAY'] = future_df['CALENDAR_DATE'].dt.day
        
        # 9. Feature Encoding for Future Data
        for col in ['DAY_OF_WEEK']:
            # Handle unseen labels in future data
            try:
                future_df.loc[:, col] = label_encoders[col].transform(future_df[col])
            except ValueError:
                logging.warning(f"Unseen label in column '{col}' during prediction. Assigning -1.")
                future_df.loc[:, col] = -1

        # 10. Prepare Future Data for Prediction
        X_future = future_df[features]
        X_future = imputer.transform(X_future)

        # 11. Predict Future Data
        predictions = {}
        for target in targets:
            predictions[target] = models[target].predict(X_future)

        # 12. Combine Predictions and Decode
        # Create a temporary DataFrame to hold the decoded predictions
        decoded_predictions = pd.DataFrame()
        for target in targets:
            # Ensure predictions are integers before inverse_transform
            predictions[target] = np.round(predictions[target]).astype(int)
            decoded_predictions.loc[:, target] = label_encoders[target].inverse_transform(predictions[target])

        # Concatenate the decoded predictions with the future_df
        future_df = pd.concat([future_df, decoded_predictions], axis=1)

        # 12.1 Fill "N/A" in SPECIAL_RUN_CD if needed
        
        if 'SPECIAL_RUN_CD' in future_df.columns:
            future_df.loc[:, 'SPECIAL_RUN_CD'] = np.where(future_df['SPECIAL_RUN_CD'] == 'N/A', 'N/A', future_df['SPECIAL_RUN_CD'])

        # 13. Reorder and Format
        future_df = future_df[['APP_TYPE', 'CYCLE_TYPE', 'CALENDAR_DATE', 'SPECIAL_RUN_CD']]
        future_df.loc[:, 'CALENDAR_DATE'] = future_df['CALENDAR_DATE'].dt.strftime('%d-%b-%y')
        logging.info(f"Prediction completed successfully for {future_year}")
        return future_df
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise

def fetch_data_from_database():
    """Fetches data from the database and saves it to a CSV file."""
    logging.info("Fetching data from the database...")
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        # Connection string
        conn_str = (
            r'DRIVER={ODBC Driver 17 for SQL Server};'  # Replace with your driver if needed
            r'SERVER=' + DATABASE_SERVER + ';'
            r'DATABASE=' + DATABASE_NAME + ';'
            r'UID=' + DATABASE_USER + ';'
            r'PWD=' + DATABASE_PASSWORD + ';'
        )

        # Establish the database connection
        cnxn = pyodbc.connect(conn_str)
        cursor = cnxn.cursor()

        # SQL query to fetch data
        query = f"SELECT * FROM {DATABASE_TABLE}"
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Get column names
        columns = [column[0] for column in cursor.description]

        # Create a DataFrame from the fetched data
        df = pd.DataFrame.from_records(rows, columns=columns)

        # Save the DataFrame to a CSV file
        output_file = os.path.join(OUTPUT_DIRECTORY, 'cycle_calender_data.csv')
        df.to_csv(output_file, index=False)
        logging.info(f"Data fetched from database and saved to: {output_file}")
        return output_file

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        logging.error(f"Database error: {sqlstate}", exc_info=True)
        raise Exception(f"Database error: {sqlstate}")
    except Exception as e:
        logging.error(f"An error occurred while fetching data from the database: {e}", exc_info=True)
        raise Exception(f"An error occurred while fetching data from the database: {e}")
    finally:
        if 'cnxn' in locals() and cnxn: # check if cnxn is defined and not None
            cnxn.close()

def generate_insert_queries(df, table_name):
    """Generates SQL INSERT queries from a DataFrame."""
    logging.info("Generating INSERT queries...")
    queries = []
    for index, row in df.iterrows():
        columns = ', '.join(row.index)
        values = ', '.join([f"'{val}'" if isinstance(val, str) else str(val) for val in row.values])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values});"
        queries.append(query)
    logging.info(f"Generated {len(queries)} INSERT queries.")
    return queries

def process_data(future_year, input_file=None):
    """
    Processes the data, generates predictions, and saves the output.

    Args:
        future_year (int): The year for which to generate predictions.
        input_file (str, optional): Path to the input file. Defaults to None.
    """
    logging.info(f"Starting data processing for year: {future_year}, input file: {input_file}")
    try:
        if input_file:
            # Load the input data
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.xlsx'):
                df = pd.read_excel(input_file)
            else:
                raise Exception("Error: Input file must be a CSV or XLSX file.")

            # Validate required columns
            required_columns = ['APP_TYPE', 'CYCLE_TYPE', 'CALENDAR_DATE', 'SPECIAL_RUN_CD']
            if not all(col in df.columns for col in required_columns):
                raise Exception(f"Error: Input file must contain the following columns: {', '.join(required_columns)}")
        else:
            logging.info("No input file provided. Fetching data from the database...")
            input_file = fetch_data_from_database()
            df = pd.read_csv(input_file)

        # Generate predictions using ML
        predicted_df = predict_future_data_ml(df, future_year)

        # Save the output to an Excel file
        output_file = f"prediction_{future_year}.xlsx"
        predicted_df.to_excel(output_file, index=False)
        logging.info(f"Prediction for {future_year} saved to {output_file}")
        
        # Generate and save INSERT queries
        insert_queries = generate_insert_queries(predicted_df, DATABASE_TABLE)
        sql_output_file = f"insert_queries_{future_year}.sql"
        with open(sql_output_file, 'w') as f:
            for query in insert_queries:
                f.write(query + '\n')
        logging.info(f"INSERT queries for {future_year} saved to {sql_output_file}")
        return output_file, sql_output_file

    except FileNotFoundError as fnfe:
        logging.error(f"File not found error during data processing: {fnfe}", exc_info=True)
        raise Exception(f"File not found: {fnfe}")
    except pd.errors.EmptyDataError as ed:
        logging.error(f"Empty data error during data processing: {ed}", exc_info=True)
        raise Exception(f"Empty data error: {ed}")
    except pd.errors.ParserError as pe:
        logging.error(f"Parser error during data processing: {pe}", exc_info=True)
        raise Exception(f"Parser error: {pe}")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}", exc_info=True)
        raise Exception(f"An error occurred: {e}")

def generate_button_clicked():
    """Handles the click event of the Generate button."""
    logging.info("Generate button clicked.")
    try:
        future_year_str = year_entry.get()
        if not future_year_str.isdigit():
            raise ValueError("Future year must be an integer.")
        future_year = int(future_year_str)

        use_file = file_checkbox_var.get()
        input_file = None
        if use_file:
            input_file = file_path_var.get()
            if not input_file:
                raise Exception("Please select a file.")
            if not os.path.exists(input_file):
                raise Exception(f"Error: Input file '{input_file}' not found.")
        
        # Run the process_data in a separate thread
        threading.Thread(target=run_process_data, args=(future_year, input_file)).start()

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        logging.error(f"Exception: {e}")
        messagebox.showerror("Error", str(e))

def run_process_data(future_year, input_file):
    """
    Runs the data processing in a separate thread and handles the result.
    """
    try:
        output_file, sql_output_file = process_data(future_year, input_file)
        messagebox.showinfo("Success", f"Prediction saved to {output_file}\nINSERT queries saved to {sql_output_file}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def browse_file():
    """Opens a file dialog to select an input file."""
    logging.info("Browse file button clicked.")
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel and CSV files", "*.xlsx *.csv"), ("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
    )
    file_path_var.set(file_path)
    logging.info(f"Selected file: {file_path}")

# Create the main window
window = tk.Tk()
window.title("Generating annual cycle calender utility")

# Title Label
title_label = ttk.Label(window, text="Generating annual cycle calender utility", font=("Arial", 16, "bold"))
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# File Input Checkbox
file_checkbox_var = tk.BooleanVar()
file_checkbox = ttk.Checkbutton(window, text="Provide input via file (Excel/CSV)", variable=file_checkbox_var)
file_checkbox.grid(row=1, column=0, columnspan=2, pady=5, sticky="w")

# File Path
file_path_var = tk.StringVar()
file_path_entry = ttk.Entry(window, textvariable=file_path_var, width=40, state="disabled")
file_path_entry.grid(row=2, column=0, padx=5, pady=5)

browse_button = ttk.Button(window, text="Browse", command=browse_file, state="disabled")
browse_button.grid(row=2, column=1, padx=5, pady=5)

def toggle_file_input():
    """Enables/disables file input based on checkbox state."""
    if file_checkbox_var.get():
        file_path_entry.config(state="normal")
        browse_button.config(state="normal")
    else:
        file_path_entry.config(state="disabled")
        browse_button.config(state="disabled")

file_checkbox.config(command=toggle_file_input)

# Prediction Year
year_label = ttk.Label(window, text="Prediction Year:")
year_label.grid(row=3, column=0, pady=5, sticky="w")

year_entry = ttk.Entry(window, width=10)
year_entry.grid(row=3, column=1, pady=5, sticky="w")

# Generate Button
generate_button = ttk.Button(window, text="Generate", command=generate_button_clicked)
generate_button.grid(row=4, column=0, columnspan=2, pady=10)

# Run the UI
window.mainloop()

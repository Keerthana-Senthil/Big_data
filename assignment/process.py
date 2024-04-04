import pandas as pd
import numpy as np
import os
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

def process_selected_columns(file_path, file_name):
    # Read selected columns from CSV file
    df = pd.read_csv(file_path)
    name= "new_" + file_name
    path = os.path.join("assignment/selected_column",name)
    if os.path.exists(path):
    # If the path exists, proceed to read the CSV file
        df2 = pd.read_csv(path)
    else:
        return
    
    
    # Convert 'DATE' column to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Iterate through columns in df2
    for col in df2.columns:
        if col in df.columns:
            if col == 'MonthlyAverageRH':
                # Compute monthly average of DailyAverageRelativeHumidity and save it in df2
                #df2[col] = df.groupby(df['DATE'].dt.month)['DailyAverageRelativeHumidity'].mean()
                df2[col] = df['DailyAverageRelativeHumidity'].astype(str).str.replace('s', '').apply(lambda x: float(x) if isinstance(x, str) and x.replace('.', '').isdigit() else np.nan).groupby(df['DATE'].dt.month).mean()
          
            elif col.startswith('MonthlyDaysWithGT001Precip'):
                # Count the monthly number of DailyPrecipitation >= 0.01 and save it in df2
                df2[col] = (df['DailyPrecipitation'].apply(lambda x: float(x.replace('s', '')) if isinstance(x, str) and x.replace('s', '').replace('.', '').isdigit() else np.nan) >= 0.01).groupby(df['DATE'].dt.month).sum()
            elif col.startswith('MonthlyDaysWithGT010Precip'):
                # Count the monthly number of DailyPrecipitation >= 0.1 and save it in df2
                df2[col] = (df['DailyPrecipitation'].apply(lambda x: float(x.replace('s', '')) if isinstance(x, str) and x.replace('s', '').replace('.', '').isdigit() else np.nan) >= 0.1).groupby(df['DATE'].dt.month).sum()
            elif col.startswith('MonthlyDaysWithGT32Temp'):
                # Count the monthly number of DailyMaximumDryBulbTemperature >= 32 and save it in df2
                df2[col] = (df['DailyMaximumDryBulbTemperature'].apply(lambda x: float(x.replace('s', '')) if isinstance(x, str) and x.replace('s', '').replace('.', '').isdigit()  else np.nan) >= 32).groupby(df['DATE'].dt.month).sum()
            elif col.startswith('MonthlyDaysWithGT90Temp'):
                # Count the monthly number of DailyMaximumDryBulbTemperature >= 90 and save it in df2
                df2[col] = (df['DailyMaximumDryBulbTemperature'].apply(lambda x: float(x.replace('s', '')) if isinstance(x, str) and x.replace('s', '').replace('.', '').isdigit()  else np.nan) >= 90).groupby(df['DATE'].dt.month).sum()
            elif col.startswith('MonthlyDaysWithLT0Temp'):
                # Count the monthly number of DailyMinimumDryBulbTemperature <= 0 and save it in df2
                df2[col] = (df['DailyMinimumDryBulbTemperature'].apply(lambda x: float(x.replace('s', '')) if isinstance(x, str) and x.replace('s', '').replace('.', '').isdigit()  else np.nan) <= 0).groupby(df['DATE'].dt.month).sum()
            elif col.startswith('MonthlyDaysWithLT32Temp'):
                # Count the monthly number of DailyMinimumDryBulbTemperature <= 32 and save it in df2
                df2[col] = (df['DailyMinimumDryBulbTemperature'].apply(lambda x: float(x.replace('s', '')) if isinstance(x, str) and x.replace('s', '').replace('.', '').isdigit()  else np.nan) <= 32).groupby(df['DATE'].dt.month).sum()
            elif col == 'MonthlyDewpointTemperature':
                # Compute monthly average of DailyAverageDewPointTemperature and save it in df2
                #df2[col] = df.groupby(df['DATE'].dt.month)['DailyAverageDewPointTemperature'].mean()
                df2[col] = df['DailyAverageDewPointTemperature'].astype(str).str.replace('s', '').apply(lambda x: float(x) if isinstance(x, str) and x.replace('.', '').isdigit() else np.nan).groupby(df['DATE'].dt.month).mean()
            elif col == 'MonthlyMeanTemperature':
                # Compute monthly average of DailyAverageDryBulbTemperature and save it in df2
                #df2[col] = df.groupby(df['DATE'].dt.month)['DailyAverageDryBulbTemperature'].mean()
                df2[col] = df['DailyAverageDryBulbTemperature'].astype(str).str.replace('s', '').apply(lambda x: float(x) if isinstance(x, str) and x.replace('.', '').isdigit() else np.nan).groupby(df['DATE'].dt.month).mean()
         
    
    # Save df2 to the same file
    df2 = df2.drop(df.index[0])
    df2.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

# Example usage:
for root, dirs, files in os.walk("assignment/data"):
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)
            process_selected_columns(file_path, file_name)

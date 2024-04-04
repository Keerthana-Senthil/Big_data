import pandas as pd
import os
import numpy as np
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

def process_files_in_folder(folder_path):
    selected_columns_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.csv'):  # Assuming files are CSV format, change the condition if needed
                file_path = os.path.join(root, file_name)
                print("Processing file:", file_path)
                df = pd.read_csv(file_path)
                
                selected_columns = [col for col in df.columns if col.startswith('Monthly')]
                selected_columns = [col for col in selected_columns if any(x in col for x in ['MonthlyAverageRH', 'MonthlyDaysWithGT001Precip', 'MonthlyDaysWithGT010Precip', 
                                                                                             'MonthlyDaysWithGT32Temp', 'MonthlyDaysWithGT90Temp', 'MonthlyDaysWithLT0Temp', 
                                                                                             'MonthlyDaysWithLT32Temp','MonthlyDewpointTemperature', 'MonthlyMeanTemperature']) and df[col].notnull().any()]
                if len(selected_columns) != 0:
                    selected_columns_dict[file_name] = selected_columns
                    # Save selected columns data to a new CSV file
                    selected_columns_df = df[selected_columns]
                    selected_columns_df.to_csv(os.path.join("assignment/selected_column", f"new_{file_name}"), index=False)
               
                    new_df = pd.DataFrame(columns=selected_columns)
                    df['DATE'] = pd.to_datetime(df['DATE'])
                    for col in selected_columns:
                        column_data = []
                        month = 1  # Reset month for each column
                        for i,ele in enumerate(df[col]):
                            if pd.notnull(ele):
                                if isinstance(ele, str) and 's' in ele:
                                    ele = ele.replace('s', '')
        
                                if not (isinstance(ele,float) or isinstance(ele,int)):
                                    ele = float(ele) if isinstance(ele, str) and ele.replace('.', '').isdigit() else np.nan
                        
                                if df['DATE'].iloc[i].month == month:
                                    column_data.append(ele)
                                    month += 1
                                else:
                                    diff = df['DATE'].iloc[i].month - month
                                    while diff > 0:
                                        column_data.append(None)
                                        diff -= 1
                                    column_data.append(ele)
                                    month = df['DATE'].iloc[i].month + 1
                        
                        new_df[col] = column_data
                    
                    new_path = os.path.join("assignment/compare_data", f"compare_{file_name}")
                    new_df.to_csv(new_path, index=False)

    return selected_columns_dict

if len(process_files_in_folder("assignment/data")) == 0:
    print("There are no monthly data in the report")

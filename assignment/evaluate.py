import os
import pandas as pd
from sklearn.metrics import r2_score
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

def compute_r_square(y_true, y_pred):
    # Combine y_true and y_pred into a DataFrame
    combined_df = pd.concat([y_true, y_pred], axis=1, keys=['y_true', 'y_pred'])
    
    # Drop rows with NaN values
    combined_df = combined_df.dropna()
    
    # Separate y_true and y_pred after dropping NaN values
    y_true = combined_df['y_true']
    y_pred = combined_df['y_pred']
    
    # Compute R-square value if both y_true and y_pred are not empty after dropping NaN values
    if not y_true.empty and not y_pred.empty:
        return r2_score(y_true, y_pred)
    else:
        return None

def calculate_overall_score():
    compare_data_folder = "assignment/compare_data"
    selected_column_folder = "assignment/selected_column"
    overall_score = []

    contents = os.listdir(compare_data_folder)
        # Check if the list is empty
    if not contents:
        return  
    # Iterate through files in compare_data folder
    for filename in os.listdir(compare_data_folder):
        if filename.startswith("compare_"):
            file_number = filename.split("_")[1]
            compare_file_path = os.path.join(compare_data_folder, filename)
            
            # Check if corresponding file exists in selected_column folder
            selected_column_file_path = os.path.join(selected_column_folder, f"new_{file_number}")
            if os.path.exists(selected_column_file_path):
                # Read data from both files
                compare_data = pd.read_csv(compare_file_path)
                selected_column_data = pd.read_csv(selected_column_file_path)
                print(f"For {file_number}")
                # Compute R-square for each column
                r_square_values = {}
                for column in compare_data.columns:
                    if column in selected_column_data.columns:
                        r_square = compute_r_square(compare_data[column].iloc[0:12], selected_column_data[column].iloc[0:12])
                        r_square_values[column] = r_square
                print(r_square_values)
                # Calculate average R-square value across all columns
                overall_score.append(sum(r_square_values.values()) / len(r_square_values))
            else:
                print(f"Corresponding file not found for {filename}")

    # Calculate overall average R-square value
        overall_average_score = sum(overall_score) / len(overall_score)

        print("Overall average R-square value:", overall_average_score)
calculate_overall_score()
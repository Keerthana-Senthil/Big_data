from datetime import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
import os
from airflow.sensors.filesystem import FileSensor
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict 
import pandas as pd
import glob
import apache_beam as beam
import numpy as np
import shutil



def extract_Data_frame(directory_path, output_dir, fields):
 # Check if directory_path exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist.")

    # Check if output_dir exists or create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def extract_monthly(date_str): #Extract themonth from the data
        date_components = date_str.split('-') + date_str.split('/') + date_str.split('T')
        for component in date_components:
            if component.isdigit() and 1 <= int(component) <= 12:
                return component
        return None

    def convert_to_float(df, fields): #Convert the field values to numeric
        for field in fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        return df

    def extract_monthly_data(data): #Making sure that the fields are stored as a listof their every month entries
        grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) #A dictionary to store latitude, longitude, monthly grouped data
        for row in data:
            lat_lon_key = (row[0], row[1])
            month = extract_monthly(row[2])
            for i, field in enumerate(fields, start=3):
                grouped_data[lat_lon_key][field][month].append(row[i])
        #Converting the dictionary to the tuple 
        final_data = [] 
        for lat_lon, field_data in grouped_data.items():
            lat, lon = lat_lon
            fieldly_data = []
            for field, monthly_values in field_data.items():
                sorted_keys = sorted(monthly_values.keys())
                month_wise = [monthly_values[key] for key in sorted_keys]
                fieldly_data.append(month_wise)
            final_data.append((lat, lon, fieldly_data)) #Returning the tuple
        return final_data

    all_files = glob.glob(directory_path + "/*.csv") #Opening all csv files in the data folder
    df_list = []
    for file in all_files:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(file)

        # Filter DataFrame based on required fields
        filtered_df = df[["LATITUDE", "LONGITUDE", "DATE"] + fields]

        df_list.append(filtered_df)

    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = convert_to_float(combined_df, fields) #Converting the fields to numeric

    with beam.Pipeline() as pipeline:
        # Group data by latitude and longitude pairs and compute monthly data
        final_data = (
            pipeline
            | "Create PCollections" >> beam.Create([combined_df.values.tolist()])
            | "Extract Monthly Data" >> beam.Map(extract_monthly_data)
        )

        # Collect the elements of PCollection into a list
        final_data_list = final_data | beam.combiners.ToList()

        # Extract the first (and only) element from the resulting list
        result = final_data_list | beam.ParDo(lambda x: x[0])

        # Write the data to a text file
        result | beam.io.WriteToText(output_dir+"/First_pipeline.txt")
    #Return the location of the text file
    return output_dir


def compute_monthly_averages(data, fields):
    lat, lon, monthly_data = data

    # Compute monthly averages for each field
    averaged_data = []
    for field_values in monthly_data:
        field_averages = [np.nanmean(month) for month in field_values]
        averaged_data.append(field_averages)

    return lat, lon, averaged_data

def compute_averages_dataframe(output_dir, fields, **kwargs):
    def parse_line(line):
    # Define the scope with 'nan' available as python can't directly prase nan
        scope = {"nan": np.nan}
        return eval(line, scope)

    ti = kwargs['ti']  # Fetch the contents returned by the previous function
    pipeline_txt = ti.xcom_pull(task_ids='extract_files_task')
    print(pipeline_txt)
    # pipeline_txt contains the path to the file
    file_path = os.path.join(pipeline_txt, os.listdir(pipeline_txt)[0])

    with beam.Pipeline() as pipeline:
        # Read the text file
        data = pipeline | beam.io.ReadFromText(file_path)

        # Process the data
        transformed_data = (
            data
            | "Parse lines" >> beam.Map(parse_line)  # Convert each line to Python object
            | "Compute monthly averages" >> beam.Map(compute_monthly_averages, fields) #Compute monthly averages
        )

        # Convert transformed data back to JSON
        transformed_data_json = transformed_data | beam.Map(lambda x: str(x))

        # Write the transformed data to a text file
        transformed_data_json | beam.io.WriteToText(output_dir + '/averages_output.txt')
    #Returning the location of the text file 
    return output_dir



def plot_data_from_file(output_dir, fields, ** kwargs):#Function to plot the average of each field
    def parse_line(line):
    # Replace 'nan' with np.nan
        line = line.replace('nan', 'np.nan')
        # Evaluate the line to get the tuple
        return eval(line)
    def preprocess_data(dataframe, fields):
    # Create a new DataFrame to store the processed data
        processed_data = dataframe.copy()
        
        # Flatten the lists of values in each field average of each month
        for field in fields:
            processed_data[field] = processed_data[field].apply(lambda x: np.nanmean(x) if isinstance(x, list) else x)
        
        return processed_data


    ti = kwargs['ti']
    pipeline_txt = ti.xcom_pull(task_ids='process_file') #Pulling the data tuples from compute average dataframe function
    file_path = os.path.join(pipeline_txt, os.listdir(pipeline_txt)[0])
    Store_data = pd.DataFrame() #Creating a pandas dataframe to store the tuple
    Store_data["LATITUDE"] = ''
    Store_data["LONGITUDE"] = ''
    for field in fields:
        Store_data[field] = ''

    with open(file_path, 'r') as file:
        for line in file:
            # Parse the line to extract latitude, longitude, and values
            latitude, longitude, values = parse_line(line)
            Store_data.loc[len(Store_data)] = [latitude, longitude] + values
           


    # Create plots based on the extracted data
    # Convert latitude and longitude to a GeoDataFrame
    # Preprocess the data to handle lists of values
    print(Store_data)
    processed_data = preprocess_data(Store_data, fields) #Taking monthly averages
    print(processed_data)
    # Convert latitude and longitude to a GeoDataFrame
    gdf = gpd.GeoDataFrame(processed_data, geometry=gpd.points_from_xy(processed_data.LONGITUDE, processed_data.LATITUDE))
    
    # Set up the plot
    for field in fields:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the heatmap
        gdf.plot(column=field, cmap='viridis', linewidth=0.8, ax=ax, legend=True)
        ax.set_title(f"Heatmap for {field}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        
        # Save the plot as a .png file
        filename = os.path.join(output_dir, f"{field}.png")
        plt.savefig(filename)
        plt.close()


def plot_data_from_file2(output_dir, fields, ** kwargs):#Function to plot monthly field plots
    def parse_line(line):
    # Replace 'nan' with np.nan
        line = line.replace('nan', 'np.nan')
        # Evaluate the line to get the tuple
        return eval(line)
 
    # Create a new DataFrame to store the processed data


    ti = kwargs['ti']
    pipeline_txt = ti.xcom_pull(task_ids='process_file') #Pulling the tuple from create average dataset function
    file_path = os.path.join(pipeline_txt, os.listdir(pipeline_txt)[0])
    Store_data = pd.DataFrame()
    Store_data["LATITUDE"] = ''
    Store_data["LONGITUDE"] = ''
    for field in fields:
        Store_data[field] = ''

    with open(file_path, 'r') as file:
        for line in file:
            # Parse the line to extract latitude, longitude, and values
            latitude, longitude, values = parse_line(line)
            Store_data.loc[len(Store_data)] = [latitude, longitude] + values
           


            # Create plots based on the extracted data
    # Convert latitude and longitude to a GeoDataFrame
        # Preprocess the data to handle lists of values
    print(Store_data)
    #Creating folder for each field in the output directory
    for field in fields:
        directory_path = os.path.join(output_dir, field)
        
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # If it doesn't exist, create it
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
            
            # Give write access to the directory
            try:
                os.chmod(directory_path, 0o777)  # 0o777 represents full permissions for owner, group, and others
                print("Write access granted to the directory.")
            except OSError as e:
                print(f"Error occurred while granting write access: {e}")

    
    
    for i in range(12):  #Creating monthly plots
        processed_data = pd.DataFrame() 
        processed_data["LATITUDE"]= Store_data['LATITUDE']
        processed_data["LONGITUDE"]= Store_data['LONGITUDE']
        # Flatten the lists of values in each field
        for field in fields:
            temp=[]
 
            for j in range(len(Store_data[field])):
                if (len(Store_data[field][j])>i):
                    temp.append(Store_data[field][j][i])
                    
                else:
                    temp.append(np.nan)
            processed_data[field] = temp #Storing only the data corresponding to a given month
            print(processed_data[field])
        
       
    # Convert latitude and longitude to a GeoDataFrame
        gdf = gpd.GeoDataFrame(processed_data, geometry=gpd.points_from_xy(processed_data.LONGITUDE, processed_data.LATITUDE))
    
    # Set up the plot
        for j,field in enumerate(fields):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot the heatmap
            gdf.plot(column=field, cmap='viridis', linewidth=0.8, ax=ax, legend=True)
            ax.set_title(f"Heatmap for {field}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')
            
           
            # Save the plot as a .png file
            filename = os.path.join(output_dir, f"{field}/month{i+1}.png")
            plt.savefig(filename)
            plt.close()

def delete_data(dir_lst): #Delete all the datafiles after processing
    for folder_path in dir_lst:
        try:
        # Iterate over all files and directories in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # Check if it's a file
                if os.path.isfile(file_path):
                    # If it's a file, delete it
                    os.unlink(file_path)
                # If it's a directory, delete it recursively
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"Contents inside '{folder_path}' deleted successfully.")
        except Exception as e:
            print(f"Error occurred while deleting contents: {e}")

#Defining the variables in the program
year=2008
fields = ["HourlyDryBulbTemperature","HourlyDewPointTemperature","HourlyPressureChange"]
zip_location =f"/home/keerthana/airflow/Results/{year}.zip"
data_store_loc = f"/home/keerthana/airflow/Results/{year}_data"
pipeline_loc_1 = "/home/keerthana/airflow/pipeline/Initialpipeline"
pipeline_loc_2 = "/home/keerthana/airflow/pipeline/Secondpipeline"
image_results = "/home/keerthana/airflow/Results/Images"
archive_bash_file = "/home/keerthana/airflow/dags/check_archive.sh"



#Defining the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 19),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    
}

dag = DAG(
    'processing_data_pipeline',
    default_args=default_args,
    description='A simple DAG to process meteorological data',
    #schedule_interval='*/1 * * * *',
    schedule_interval=None,
)

#Fielsensor to verify zip location 
wait_for_archive = FileSensor(
    task_id='wait_for_archive',
    filepath=zip_location,  # Path to the archive file
    timeout=5,  # Timeout in seconds
    poke_interval=1,  # Check every second
    retries=0,  # Retry indefinitely
    mode='poke',
    dag=dag,
)

#Bash operator to verify the zip and unzip it to the location
check_validity_and_unzip = BashOperator(
    task_id= 'check_validity_and_unzip',
    bash_command=f'{archive_bash_file} {zip_location} {data_store_loc}', # A seperate bash file to unzip the file in the destined location
    dag=dag,
)

#Python function to extract the file in the location
extract_data_files = PythonOperator(
    task_id='extract_files_task',
    python_callable=extract_Data_frame,
    op_kwargs={'directory_path': data_store_loc ,"output_dir":pipeline_loc_1,"fields":fields},
    provide_context=True,
    do_xcom_push=True,
    dag=dag,
)

#Compute the monthly averages of the fields
compute_data_files = PythonOperator(
    task_id='process_file',
    python_callable=compute_averages_dataframe,
    op_kwargs={"output_dir":pipeline_loc_2 ,"fields":fields},
    provide_context=True,
    do_xcom_push=True,
    dag=dag,
)


#Plot the fields geomaps
plot_data_files = PythonOperator(
    task_id='plot_data',
    python_callable=plot_data_from_file2,
    op_kwargs={"output_dir":image_results ,"fields":fields},
    provide_context=True,
    do_xcom_push=True,
    dag=dag,

)

#Delete the csv files
delete_data_files = PythonOperator(
    task_id='delete_data_file',
    python_callable=delete_data,
    op_kwargs={"dir_lst":[data_store_loc,pipeline_loc_1,pipeline_loc_2]},
    provide_context=True,
    do_xcom_push=True,
    dag=dag,

)

#Precedence order for the tasks
wait_for_archive>>check_validity_and_unzip>>extract_data_files>>compute_data_files>>plot_data_files>>delete_data_files




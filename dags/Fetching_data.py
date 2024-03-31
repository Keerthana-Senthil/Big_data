from datetime import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
import random
import os
import zipfile
import requests
from shutil import move
from bs4 import BeautifulSoup

def select_random_files(num_files, web_page):
    # checks are added to ensure that the number of files requested is greater than zero, the HTML file exists, and the number of requested files does not exceed the available files on the webpage.
    if num_files <= 0:
        raise ValueError("Number of files must be greater than zero.")
    
    if not os.path.isfile(web_page):
        raise FileNotFoundError("Web page HTML file not found.")

    # Parse the HTML content of the webpage
    with open(web_page, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all 'a' tags with 'href' attribute
    links = soup.find_all('a', href=True)

    # Filter links to files
    files_list = [link['href'] for link in links if link['href'].endswith(('.csv'))]

    if len(files_list) < num_files:
        raise ValueError("Number of files requested exceeds the available files on the webpage.")

    # Randomly select a specified number of files
    selected_files = random.sample(files_list, num_files)
    os.remove(web_page)

    return selected_files

def fetch_data_files(url, **kwargs):
    ti = kwargs['ti'] # Fetch the contents returned by the previous function
    files_list = ti.xcom_pull(task_ids='select_data')

    file_loc = [] # List to store the locations where the file is downloaded locally
    
    for file_name in files_list:
        file_url = f"{url}/{file_name}" # The webpage url for each file
        
        print("Downloading file:", file_url)
        
        response = requests.get(file_url)
        
        # Check if the request was successful
        if response.ok:
            # Write the contents of the file to a local file
            local_file_path = os.path.basename(file_name)
            with open(local_file_path, 'wb') as f:
                f.write(response.content)
            file_loc.append(local_file_path)
            print("File downloaded and contents saved:", local_file_path)
        else:
            print("Failed to download file:", file_url)
    
    return file_loc

def zip_files(year, **kwargs):
    ti = kwargs['ti'] # Pulling the local file location returned in fetch data files function
    files_list = ti.xcom_pull(task_ids='fetch_data')

    archive_name=f"{year}.zip"
    
    # Zip the files together
    with zipfile.ZipFile(archive_name, 'w') as zipf:
        for file_path in files_list:
            zipf.write(file_path, os.path.basename(file_path))
    
    # Delete the original files downloaded locally
    for file_path in files_list:
        os.remove(file_path)

    return archive_name

def place_archive(archive_path, **kwargs):
    ti = kwargs['ti']
    archive_name = ti.xcom_pull(task_ids='zip_data')
    
    move(archive_name, archive_path) # Placing the file in the desired location 
    
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 19),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Variables in the code
year = 2008
num_of_files = 8
url = f"https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}"
destination_loc = "/home/keerthana/airflow/Results"
html_file = f"/home/keerthana/airflow/Results/{year}data.html"

# Define the DAG
dag = DAG(
    'Fetch_data_pipeline',
    default_args=default_args,
    description='A simple DAG to fetch meteorological data',
    schedule_interval=None,
)

fetch_data_url = BashOperator (
    task_id='fetch_url',
    bash_command=f"wget -O {html_file} {url}", # Bash to get website and store it as an HTML file
    do_xcom_push=True,
    dag=dag,
)

select_data_files = PythonOperator(
    task_id='select_data',
    python_callable=select_random_files,
    op_kwargs={'num_files': num_of_files, "web_page": html_file}, # Selecting random number of files
    provide_context=True,
    do_xcom_push=True,
    dag=dag,
)

fetch_files = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_files, # Download the files required
    op_kwargs={"url":url},
    provide_context=True,
    do_xcom_push=True,
    dag=dag,
)

zip_the_files = PythonOperator(
    task_id='zip_data',
    python_callable=zip_files,
    op_kwargs={'year': str(year)}, # Zip the files together
    provide_context=True,
    do_xcom_push=True,
    dag=dag,
)

place_the_files = PythonOperator(
    task_id='place_data',
    python_callable=place_archive,
    op_kwargs={'archive_path': destination_loc}, # Place the zip in the destined location
    provide_context=True,
    dag=dag,
)

# Precedence order for execution
fetch_data_url >> select_data_files >> fetch_files >> zip_the_files >> place_the_files

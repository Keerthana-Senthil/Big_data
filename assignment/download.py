import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import wget
import yaml
import io

import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 
  



def fetch_params_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    year = params.get('year')
    n_loc = params.get('n_loc')
    return year, n_loc

def fetch_and_zip_data_from_yaml(year, num_files):
    # Read parameters from YAML file
    url = f"https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}"
    html_file = f"{year}data.html"
    path = os.path.join(os.getcwd(),"assignment")
    html_path = os.path.join(path, html_file)

    # Fetch webpage
    wget.download(url, html_path)

    # Parse HTML and select files based on non-null values in "Monthly" columns
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a', href=True)

    # Reverse the order of the links
    links.reverse()

    selected_files = []

    for link in links:
        if link['href'].endswith('.csv'):
            file_url = f"{url}/{link['href']}"
            print("Checking file:", file_url)
            response = requests.get(file_url)

            if response.ok:
                # Create a file-like object from the content of the CSV file
                content = response.content
                content_file = io.BytesIO(content)

                df = pd.read_csv(content_file)
                # Check if any "Monthly" column has at least one non-null value
                monthly_columns = [col for col in df.columns if col.startswith("Monthly")]
                if any(df[col].notnull().any() for col in monthly_columns):
                    selected_files.append(link['href'])
                    print("File selected:", link['href'])
                else:
                    print("Skipping file:", link['href'])
            else:
                print("Failed to download file:", file_url)

            if len(selected_files) >= num_files:
                break

    # Fetch selected files
    file_locs = []
    for file_name in selected_files:
        file_url = f"{url}/{file_name}"
        print("Downloading file:", file_url)
        response = requests.get(file_url)
        file_folder = os.path.join(os.getcwd(),"assignment/data")
        if response.ok:
            local_file_path = os.path.join(file_folder, os.path.basename(file_name))
            with open(local_file_path, 'wb') as f:
                f.write(response.content)
            file_locs.append(local_file_path)
            print("File downloaded and saved:", local_file_path)
        else:
            print("Failed to download file:", file_url)
    
    os.remove(html_path)

# Example usage
yaml_file = "assignment/params.yaml"
year, n_loc = fetch_params_from_yaml(yaml_file)
fetch_and_zip_data_from_yaml(year, n_loc)

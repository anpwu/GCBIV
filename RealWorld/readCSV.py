import os
import pandas as pd

def list_csv_files(directory):
    csv_files = []  
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files

# Directory where CSV files are located
directory = "./Data/report/"
csv_files_list = list_csv_files(directory)
print(csv_files_list)

# Iterate over each CSV file in the list
for dataDir_ in csv_files_list:
    dataDir = directory + dataDir_
    print(f'Path: {dataDir_}')
    df = pd.read_csv(dataDir, index_col=0)
    df = df.abs()
    df_concat = pd.concat([df.mean(0), df.std(0)], axis=1)
    df_concat = df_concat.T.round(4)

    print(df_concat)

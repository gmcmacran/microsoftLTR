###################################
# Overview
# 
# This scripts performs all the data
# cleaning for Microsoft's 30K dataset.
###################################

# %%
import pandas as pd
import polars as pl
import numpy as np

##############
# Space seperated flat file to cleaned ipc file
##############
# %%
# Steps:
# 1: Load .txt files
# 2: Name columns.
# 3: Remove a few columns.
# 4: Remove leading string in values.
# 5: Convert variables to numerics.
# 6: Write ipc files for quick loads.
def create_ipc_files(filenames):

    dfs = []
    for filename in filenames:
        loadme = 'S:/Python/projects/microsoftLTR/data/' + filename + '.txt'

        temp_df = pd.read_csv(loadme, delimiter=" ", header = None)
        temp_df = pl.from_pandas(temp_df)

        dfs.append(temp_df)
    
    # variable names
    # See https://www.microsoft.com/en-us/research/project/mslr/
    for i in range(0, len(dfs)):
        df = dfs[i]
        CN = ['label', 'qid'] + ['F' + str(i-1) for i in np.arange(2, df.shape[1])]
        df.columns = CN
        dfs[i] = df
    del CN

    for i in range(0, len(dfs)):
        df = dfs[i]
        columns_to_remove = [41, 42, 43, 44, 45, 66, 67, 68, 69, 70,
                         91, 92, 93, 94, 95, 16, 17, 18, 19, 20,
                         71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                         138]
        columns_names_to_remove = []
        for j in columns_to_remove:
            columns_names_to_remove.append(df.columns[j])
        df = df.drop(columns=columns_names_to_remove)
        dfs[i] = df

    # Get ride of leading string
    for i in range(0, len(dfs)):
        df = dfs[i]
        for col in df[:,1:].columns:
            df = df.with_column(pl.col(col).str.split(":").arr.slice(1).flatten())
        dfs[i] = df
            
    # Convert to numeric
    for i in range(0, len(dfs)):
        df = dfs[i]
        for col in df.columns:
            df = df.with_column(pl.col(col).cast(pl.Float64))
        dfs[i] = df

    for i in range(0, len(dfs)):
        filename = filenames[i]
        df = dfs[i]

        writme  = 'S:/Python/projects/microsoftLTR/data/' + filename + '.ipc'
        df.write_ipc(writme)

create_ipc_files(['train', 'vali', 'test'])
from pathlib import Path
from collections import defaultdict
import pandas as pd
import glob
import os
# --- You already have this. Keeping as a placeholder here. ---
# def read_data(file_path: str) -> pd.DataFrame:
#     # return a DataFrame with a 'timestep' column + numeric columns to average
#     ...
base_dir = os.path.dirname(os.path.abspath(__file__))
def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def read_data_return_df(file_path: str) -> pd.DataFrame:
    file_paths = glob.glob(f'{file_path}/output_run*.txt')
    # print(f" --- path= {file_path}, fp = {file_paths}")
    data_frames1 = [read_data(file) for file in file_paths]
    # Merge data frames on 'timestep'
    merged_data1 = pd.concat(data_frames1).groupby('timestep').mean()
    return merged_data1

def process_root(root_dir: str):
    root = Path(root_dir)
    groups = defaultdict(list)

    # Find all matching files and group them by their parent folder
    for txt in root.rglob('output_run*.txt'):
        if txt.is_file():
            groups[txt.parent].append(txt)
    no_of_folders = len(groups)
    if not groups:
        print("No matching files found.")
        return
    folders = [str(folder) for folder,_ in groups.items()]
    print(f"folder= {folders}")
    for folder in folders:
        print(f"processing {folder}")
        my_data = read_data_return_df(folder)
        out_path_file = f"{folder}/output_data.pkl"
        my_data.to_pickle(out_path_file)
        print(f"Saved: {out_path_file}")
        no_of_folders -= 1
        print(f"** Folders left = {no_of_folders} **")

    # for folder,_ in groups.items():
    #     try:
    #         print(f"Processing {folder}")
    #         df = read_data(str(folder))
    #         out_path_file = f"{folder}/output_data.pkl"
    #         # df.to_pickle(out_path_file)
    #
    #         # Save alongside the inputs (pickle)
    #         # out_path = folder / 'merged_data.pck'
    #         # merged.to_pickle(out_path)
    #
    #         print(f"Saved: {out_path_file}")
    #     except Exception as e:
    #         pass
    #         # print(f"Failed in {folder}: {e}")

process_root(f"{base_dir}/output_data")
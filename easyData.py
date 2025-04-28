import pandas as pd
import glob as glob

# Loads all .json files from a given path into a single DataFrame
def load_jsons(path):
    dfs = []
    files = glob.glob(path)
    for file in files:
        if(".json" in file):
            tempdata = pd.read_json(file, lines=True)
            dfs.append(tempdata)
    data = pd.concat(dfs, ignore_index = True)
    return data

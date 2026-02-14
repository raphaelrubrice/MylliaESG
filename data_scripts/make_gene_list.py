import os
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent
    data_folder = root_folder / "data"
    data_path = data_folder / "sample_submission.csv"

    print("\nLoading sample submissions..")
    csv = pd.read_csv(str(data_path))
    genes = [g for g in csv.columns if g != 'pert_id']
    
    print("Writing gene list..")
    path_list = data_folder / "gene_list.txt"
    with open(path_list, "w") as f:
        for g in genes:
            f.write(g + "\n")
    print("Done.\n")
        
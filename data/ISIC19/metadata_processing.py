import pandas as pd

metadata = pd.read_csv("ISIC_2019_Training_Metadata.csv")
gt = pd.read_csv("ISIC_2019_Training_GroundTruth.csv")

metadata["diagnostic"] = ""

for i in range(len(metadata)):
    print(metadata.iloc[i, metadata.columns.get_loc("img_id")])
    if metadata.iloc[i, metadata.columns.get_loc("img_id")] == gt.iloc[i, gt.columns.get_loc("img_id")]:
        if gt.iloc[i, gt.columns.get_loc("MEL")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "MEL"
        elif gt.iloc[i, gt.columns.get_loc("NV")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "NV"
        elif gt.iloc[i, gt.columns.get_loc("BCC")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "BCC"
        elif gt.iloc[i, gt.columns.get_loc("AK")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "AK"
        elif gt.iloc[i, gt.columns.get_loc("BKL")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "BKL"   
        elif gt.iloc[i, gt.columns.get_loc("DF")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "DF"      
        elif gt.iloc[i, gt.columns.get_loc("VASC")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "VASC"   
        elif gt.iloc[i, gt.columns.get_loc("SCC")] == 1:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "SCC"
        else:
            metadata.iloc[i, metadata.columns.get_loc("diagnostic")] = "UNK"           

metadata.to_csv("processed_metadata.csv", index=False)
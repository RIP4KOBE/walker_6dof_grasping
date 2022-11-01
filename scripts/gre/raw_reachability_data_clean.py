import pandas as pd
import numpy as np
from pathlib import Path
from gpn.io import *
from gpn.perception import *
from gpn.utils.transform import Rotation, Transform


#Inspection:
#Compute the number of positive and negative samples in the dataset.
root = Path("/home/walker2/walker_ik_solver_ws/src/walker_kinematics_solver/data")
df = pd.read_csv(root / "raw/walker_reachability_data_s4_quternion(setp=0.1).csv", delim_whitespace=True)#"delim_whitespace=True" indicates that the separator is a blank character(eg,the separator in reachability.csv is ' ',so we need to set delim_whitespace=True, otherwise it will cause error )
positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]
print("Number of samples:", len(df.index))
print("Number of positives:", len(positives.index))
print("Number of negatives:", len(negatives.index)) 

#Balance
#Discard a subset of negative samples to balance classes.
positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]
i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
df = df.drop(i)

df.to_csv(root / "clean/walker_reachability_data_s4_quternion(setp=0.1)_clean.csv", index=False)
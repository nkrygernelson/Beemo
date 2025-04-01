import pandas as pd
import json
import os
#pbe, hse, scan, gllbsc, ordered_experiment, disordered_experiment
data_path = "data/band_gap_no_structs.json"
mpid_to_formula_path = "data/mpid_to_formula.json"
fidelity_path = "data/fidelity.json"
with open(data_path, "r") as f:
    data = json.load(f)

with open(mpid_to_formula_path, "r") as f:
    mpid_to_formula = json.load(f)

fidelity_dict = {"pbe":0, "hse":2, "scan":3, "gllb-sc":1, "ordered_exp":4, "disordered_exp":4}
methods = ["pbe", "hse", "scan", "gllb-sc",]
new_data = []
count = 0
for method in methods:

    method_data = data[method]
    for mp_id, band_gap in method_data.items():
        try:
            formula = mpid_to_formula[mp_id]
            formula = mpid_to_formula[mp_id]
            new_data.append([mp_id, formula, band_gap, fidelity_dict[method]])
        except:
            count += 1
            continue
        
df = pd.DataFrame(new_data, columns=["mp_id","formula", "band_gap", "fidelity"])
df.to_csv("data/fidelity_data.csv", index=False)
print(count)
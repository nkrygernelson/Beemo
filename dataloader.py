from mp_api.client import MPRester
import pandas as pd
from pymatgen.core.composition import Composition as pmg_Composition


####
#Data
ele_id_dict = {"H":1, "He":2}
def get_atomic_number_from_ele(ele):
   return pmg_Composition(ele).elements[0].Z
def formula_to_id_seq(formula):
    ele_dict = pmg_Composition(formula).get_el_amt_dict()
    seq = []
    for ele in ele_dict.keys():
        seq+=[get_atomic_number_from_ele(ele)]*int(ele_dict[ele])
    return seq




df = pd.read_csv("data.csv")

df = df[~df["formula"].isin(["nan","NaN", "NAN"])].dropna(subset=["formula"])
#if the formula contains "NaN" replace it with "Na1N"
df['formula'] = df['formula'].apply(lambda x: str(x).replace("NaN", "Na1N"))
bgs = df["BG"]
formulas = df["formula"]
seqs = []
for formula in formulas:
    seq = formula_to_id_seq(formula)
    seqs.append(seq)


'''
# Replace with your Materials Project API key
API_KEY = "GPPXFjpPCxMe6U8Uz9CQa4TEhNbaaPCY"
MP_ID = "mp-1234"  # Replace with the Materials Project ID of the material you want
def fetch_with_id(id):
    with MPRester(API_KEY) as mpr:
        # Query the material data
        data = mpr.summary.search(criteria={"task_id": MP_ID}, properties=["composition"])
        
        if data:
            # Extract composition as a dictionary of elements and their proportions
            composition = data[0]["composition"]
            element_dict = composition.get_el_amt_dict()
            
            return element_dict
        return None

data = pd.read_json("band_gap_no_structs.json")
data = data.reset_index()

seqs = []
for mpid in data["index"]:
    el_dict = fetch_with_id(mpid)
    seqs.append(el_dict)
print()
'''


from mp_api.client import MPRester
import json
api_key = "Pg7yQJaFuQOgCcqaZO9A73I1KRRdajEv"
path = "data/band_gap_no_structs.json"

with open(path, "r") as f:
    data = json.load(f)
mpid_pbe = list(data['pbe'].keys())

# Option 1: Pass your API key directly as an argument.
with MPRester(api_key) as mpr:
    docs = mpr.materials.search(
        fields=["material_id","formula_pretty", "structure"],
    )

mpid_to_formula = {}
formula = []
structure = []
for doc in docs:
   mpid_to_formula[doc.material_id] = doc.formula_pretty
   doc.structure.to("data/cif_structures/"+doc.material_id+".cif","cif")

with open("data/mpid_to_formula.json", "w") as f:
    json.dump(mpid_to_formula, f)

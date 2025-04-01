from mp_api.client import MPRester
import pandas as pd

api_key = "Pg7yQJaFuQOgCcqaZO9A73I1KRRdajEv"  # Your API key

# Define fidelity mapping based on functional
FIDELITY_MAPPING = {
    'GGA': 'low',
    'GGA+U': 'medium',
    'SCAN': 'high',
    'HSE': 'very_high',
    # Add other functionals as needed
}

def get_multi_fidelity_data(material_ids):
    with MPRester(api_key) as mpr:
        # Get materials data with task IDs and run types
        materials_docs = mpr.materials.search(
            material_ids=material_ids,
            fields=["material_id", "formula_pretty", "task_ids", "run_types"]
        )
        
        # Create mappings
        formula_map = {}
        task_run_maps = {}
        
        for doc in materials_docs:
            formula_map[doc.material_id] = doc.formula_pretty
            task_run_maps[doc.material_id] = {
                task_id: doc.run_types[task_id].value 
                for task_id in doc.task_ids 
                if task_id in doc.run_types
            }

        # Get electronic structure data with task IDs
        es_docs = mpr.electronic_structure.search(
            material_ids=material_ids,
            fields=["material_id", "task_id", "band_gap"]
        )

    # Compile data
    data = []
    for doc in es_docs:
        material_id = doc.material_id
        task_id = str(doc.task_id)  # Convert MPID to string
        
        fidelity = FIDELITY_MAPPING.get(
            task_run_maps.get(material_id, {}).get(task_id, "unknown"), 
            "unknown"
        )
        
        data.append({
            "mp-id": material_id,
            "formula": formula_map.get(material_id, ""),
            "bandgap": doc.band_gap,
            "fidelity_level": fidelity,
            "literl":task_run_maps.get(material_id, {}).get(task_id, "unknown")
        })

    return pd.DataFrame(data)

# Example usage
material_ids = ["mp-2019", "mp-19019"]  # Add your materials
df = get_multi_fidelity_data(material_ids)
print(df)
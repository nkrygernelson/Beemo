import json
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core import Composition
from collections import Counter
from mp_api.client import MPRester
def parse_gllbsc_json(json_file):

    def json_to_csv(input_file, output_file='materials_data.csv'):
        """
        Convert JSON file with materials data to CSV
        
        Parameters:
        -----------
        input_file : str
            Path to input JSON file
        output_file : str
            Path to output CSV file
        """
        # Read JSON data from file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract the relevant data and flatten it
        flattened_data = []
        
        for item in data:
            entry = {}
            
            # Basic metadata
            entry['formula'] = item.get('formula', '')
            entry['id'] = item.get('id', '')
            entry['identifier'] = item.get('identifier', '')
            entry['is_public'] = item.get('is_public', '')
            entry['last_modified'] = item.get('last_modified', '')
            entry['project'] = item.get('project', '')
            
            # Extract nested energy gap data if available
            if 'data' in item and '\u0394E' in item['data']:
                delta_e = item['data']['\u0394E']
                
                # KS (Kohn-Sham) data
                if 'KS' in delta_e:
                    if 'indirect' in delta_e['KS']:
                        entry['KS_indirect_eV'] = delta_e['KS']['indirect'].get('value', '')
                    
                    if 'direct' in delta_e['KS']:
                        entry['KS_direct_eV'] = delta_e['KS']['direct'].get('value', '')
                
                # QP (Quasiparticle) data
                if 'QP' in delta_e:
                    if 'indirect' in delta_e['QP']:
                        entry['QP_indirect_eV'] = delta_e['QP']['indirect'].get('value', '')
                    
                    if 'direct' in delta_e['QP']:
                        entry['QP_direct_eV'] = delta_e['QP']['direct'].get('value', '')
            
            # Extract C value if available
            if 'data' in item and 'C' in item['data']:
                entry['C_value'] = item['data']['C'].get('value', '')
            
            flattened_data.append(entry)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")
        
        return df


    # Specify your input and output files
    input_file = "data/src/gllbsc.json"  # Change this to your JSON file path
    output_file = "materials_data.csv"

    # Process the data
    df = json_to_csv(input_file, output_file)

    # Display information about the DataFrame
    print("\nDataFrame Information:")
    print(f"Total entries: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
    df.to_csv("data/src/gllbsc.csv", index=False)

def parse_snumat_json(json_file):
    df = pd.read_json(json_file)
    print(df.head(1))
    print(df.columns.tolist())
    def formula_from_list(formula_list):
        #list of atoms to fromula
        # H2O -> [H, H, O]
        formula_list = formula_list['elements']
        formula_dict = {}
        for atom in formula_list:
            if atom in formula_dict:
                formula_dict[atom] += 1
            else:
                formula_dict[atom] = 1
        # Convert to string
        formula_str = ''.join([f"{atom}{count}" for atom, count in formula_dict.items()])
        return formula_str
    #apply formula_from_list to the atoms column
    #full first row of df
    print(df.iloc[0])
    print("sss")
    print(df['atoms'][0].keys())
    print(df['atoms'][0]['elements'])
    #print(formula_from_list(df['atoms'][0]['elements']))

    df_new = pd.DataFrame(columns = ["snumat-id", "formula", "HSE", "GGA"])
    df_new["formula"]= df['atoms'].apply(formula_from_list)
    df_new["snumat-id"]= df['SNUMAT_id']
    df_new["HSE"]= df['Band_gap_HSE']
    df_new["GGA"]= df['Band_gap_GGA']
    df_new.to_csv("data/src/snumat.csv", index=False)
    
def parse_exp_json():
    api_key = "Pg7yQJaFuQOgCcqaZO9A73I1KRRdajEv"
    with MPRester(api_key) as mpr:
        mp_docs = mpr.materials.summary.search(fields=["material_id", "database_IDs", "formula_pretty"])

    icsd_to_mpid = {}
    for mp_doc in mp_docs:
        mpid = str(mp_doc.material_id)
        for icsd_id in mp_doc.database_IDs.get("icsd",[]):
            if icsd_id not in icsd_to_mpid:
                icsd_to_mpid[icsd_id] = []
            icsd_to_mpid[icsd_id].append(mpid)
            print(mpid, icsd_id)
    #save mpid_to_icsd as json
    with open("data/mpid_to_icsd.json", "w") as f:
        json.dump(icsd_to_mpid, f)


#parse_snumat_json("data/src/snumat.json") 
#parse_gllbsc_json("data/src/gllbsc.json")
parse_exp_json()
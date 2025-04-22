import pandas as pd
import numpy as np
import pymatgen

#files to read from
expt = 'data/csv/exp_data.csv'
gllbsc = 'data/csv/gllbsc.csv'
PBESol_SCAN = 'data/csv/PBE_SCAN.csv'
HSE = 'data/csv/snumat.csv'
MP = 'data/csv/MP_functional.csv'

mapping = {
        'GGA': 0,
        'PBEsol': 0,
        'GGA+U': 1,
        'r2SCAN': 2,
        'SCAN': 2,
        'GLLBSC': 3,
        'HSE': 4,
        'EXPT': 5,
        }

def expt_data():
    """
    Read experimental data from a CSV file.
    
    Args:
        expt (str): Path to the CSV file containing experimental data.
        
    Returns:
        pd.DataFrame: DataFrame containing the experimental data.
    """
    df_expt = pd.read_csv(expt)
    df = pd.DataFrame()
    df['formula'] = df_expt['formula']
    df['BG'] = df_expt['expt_gap']
    fidelity = [mapping['EXPT']]*len(df_expt)
    df['fidelity'] = fidelity
    df['fidelity'] = df['fidelity'].astype(int)
    return df

def gllbsc_data():
    df_gllbsc = pd.read_csv('data/csv/gllbsc.csv')
    df = pd.DataFrame()
    df['formula'] = df_gllbsc['formula']
    df['BG'] = np.minimum(df_gllbsc['QP_indirect'], df_gllbsc['QP_direct'])
    #divide by 1000
    df['BG'] = df['BG']/1000
    fidelity = [mapping['GLLBSC']]*len(df_gllbsc)
    df['fidelity'] = fidelity
    df['fidelity'] = df['fidelity'].astype(int)
    return df

def pbe_scan_data():
    df_pbe_scan = pd.read_csv(PBESol_SCAN)
    print(df_pbe_scan.columns)
    df_scan = pd.DataFrame()
    df_pbesol = pd.DataFrame()
    df_scan['formula'] = df_pbe_scan['formula']
    df_pbesol['formula'] = df_pbe_scan['formula']
    #PBE_sol_bg, SCAN_bg,
    df_scan["BG"] = df_pbe_scan['SCAN_bg']
    df_pbesol["BG"] = df_pbe_scan['PBE_sol_bg']
    fidelity_scan = [mapping['SCAN']]*len(df_scan)
    fidelity_pbesol = [mapping['GGA']]*len(df_pbesol)
    df_scan['fidelity'] = fidelity_scan
    df_pbesol['fidelity'] = fidelity_pbesol
    df_scan['fidelity'] = df_scan['fidelity'].astype(int)
    df_pbesol['fidelity'] = df_pbesol['fidelity'].astype(int)
    return df_scan, df_pbesol
def hse_data():
    df_snumat = pd.read_csv('data/csv/snumat.csv')
    df_hse = pd.DataFrame()
    df_gga = pd.DataFrame()
    df_hse['formula'] = df_snumat['formula']
    df_gga['formula'] = df_snumat['formula']
    df_hse['BG'] = df_snumat['HSE']
    df_gga['BG'] = df_snumat['GGA']
    fidelity_hse = [mapping['HSE']]*len(df_hse)
    fidelity_gga = [mapping['GGA']]*len(df_gga)
    df_hse['fidelity'] = fidelity_hse
    df_gga['fidelity'] = fidelity_gga
    df_hse['fidelity'] = df_hse['fidelity'].astype(int)
    df_gga['fidelity'] = df_gga['fidelity'].astype(int)
    return df_hse, df_gga

def mp_data():
    df_mp = pd.read_csv(MP)
    df = pd.DataFrame()
    #discard "uknown" rows
    df_mp = df_mp[df_mp['Functional'] != 'unknown']
    df['formula'] = df_mp['formula']
    df['BG'] = df_mp['bandgap']
    df['fidelity'] = df_mp['Functional'].apply(lambda x: mapping[x])
    df['fidelity'] = df['fidelity'].astype(int)
    #we need to separate the GGA, GGA+U, r2Scan, SCAN
    #sepearate by fiedlity into three dataframes 
    df_gga = df[df['fidelity'] == 0]
    df_ggau = df[df['fidelity'] == 1]
    df_scan = df[df['fidelity'] == 2]
    df_gllbsc = df[df['fidelity'] == 3]
    return df_gga, df_ggau, df_scan, df_gllbsc
    
df_mp_gga, df_mp_ggau, df_mp_scan, df_mp_gllbsc = mp_data()
df_sol_scan, df_pbesol = pbe_scan_data()
df_hse, df_hse_gga = hse_data()
df_expt = expt_data()
df_gllbsc = gllbsc_data()

#group by functional
df_GGA = pd.concat([df_mp_gga,df_hse_gga, df_pbesol])
df_GGAU = df_mp_ggau
df_SCAN = pd.concat([df_mp_scan, df_sol_scan])
df_GLLBSC = pd.concat([df_mp_gllbsc, df_gllbsc])
df_HSE = df_hse
df_EXPT = df_expt
#remove duplicates of the whole row
df_GGA = df_GGA.drop_duplicates()
df_GGAU = df_GGAU.drop_duplicates()
df_SCAN = df_SCAN.drop_duplicates()
df_GLLBSC = df_GLLBSC.drop_duplicates()
df_HSE = df_HSE.drop_duplicates()
df_EXPT = df_EXPT.drop_duplicates()
#list how many duplicates there are on the formula subset
print("GGA duplicates: ", df_GGA.duplicated(subset=['formula']).sum())
print("SCAN duplicates: ", df_SCAN.duplicated(subset=['formula']).sum())
print("GLLBSC duplicates: ", df_GLLBSC.duplicated(subset=['formula']).sum())
print("HSE duplicates: ", df_HSE.duplicated(subset=['formula']).sum())
print("EXPT duplicates: ", df_EXPT.duplicated(subset=['formula']).sum())

#save them to csv

df_GGA.to_csv('data/train/GGA.csv', index=False)
df_GGAU.to_csv('data/train/GGAU.csv', index=False)
df_SCAN.to_csv('data/train/SCAN.csv', index=False)
df_GLLBSC.to_csv('data/train/GLLBSC.csv', index=False)
df_HSE.to_csv('data/train/HSE.csv', index=False)
df_EXPT.to_csv('data/train/EXPT.csv', index=False)

#Merge all dataframes into one
df_all = pd.concat([df_GGA, df_SCAN, df_GLLBSC, df_HSE, df_EXPT])
df_all.to_csv('data/train/all_data.csv', index=False)




'''
HSE_data, GGA_data = hse_data()
GLLBSC_data = gllbsc_data()
EXPT_data = expt_data()
scan_data, PBEsol_data = pbe_scan_data()
MP_data = mp_data()
HSE_data.to_csv('data/train/HSE.csv', index=False)
GLLBSC_data.to_csv('data/train/GLLBSC.csv', index=False)
EXPT_data.to_csv('data/train/EXPT.csv', index=False)
'''


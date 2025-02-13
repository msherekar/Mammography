import pandas as pd
import os
import argparse

def data_sampling(in_data, subs):
    sort_data = in_data.copy()
    sort_data.drop_duplicates(subset="patient_id", ignore_index=True, inplace=True) 
    print(sort_data.head())
    patients = pd.DataFrame()
    #equally sampling in each subgroup combinations
    cancer_list = [0, 1]
    density_list = [1, 2, 3, 4]
    race_list = ["African American  or Black", "Caucasian or White", "other"]
    for i in cancer_list:
        for j in density_list:
            for k in race_list:
                if k != "other":
                    data_temp = sort_data[(sort_data['cancer'] == i)&(sort_data['tissueden'] == j)&(sort_data['ETHNICITY_DESC'] == k)]
                else:
                    data_temp = sort_data[(sort_data['cancer'] == i)&(sort_data['tissueden'] == j)&(sort_data['ETHNICITY_DESC'] != "African American  or Black")&(sort_data['ETHNICITY_DESC'] != "Caucasian or White")]
                data_temp = data_temp.sample(frac=subs)
                patients = patients.append(data_temp)
    print(patients.head())           
    out_data = in_data.loc[in_data['patient_id'].isin(patients['patient_id'])]
    
    return out_data     

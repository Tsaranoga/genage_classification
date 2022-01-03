""" Gwendolyn Gusak, Tobias Woertwein """

'''
Completely obsolete as everything is now done in 'GENage_data_preprocessing.py'.
'''

import pandas as pd

if __name__ == '__main__':

    # 1. Import the csv files: env.csv, pphen.csv, psite.csv

    #   a) environment file
    col_names1 = pd.read_csv('./data/env.csv', nrows=0).columns
    dtypes_env = {'Unnamed: 0': str}
    dtypes_env.update({col: int for col in col_names1 if col not in dtypes_env})
    environment = pd.read_csv('./data/env.csv', dtype=dtypes_env)

    #   b) plant-associated phenotype file (path/symb)
    col_names2 = pd.read_csv('./data/pphen.csv', nrows=0).columns
    dtypes_pphen = {'Unnamed: 0': str}
    dtypes_pphen.update({col: int for col in col_names2 if col not in dtypes_pphen})
    pa_pheno = pd.read_csv('./data/pphen.csv', dtype=dtypes_pphen)

    #   c) plant-association site file
    col_names3 = pd.read_csv('./data/psite.csv', nrows=0).columns
    dtypes_psite = {'Unnamed: 0': str}
    dtypes_psite.update({col: int for col in col_names3 if col not in dtypes_psite})
    pa_site = pd.read_csv('./data/psite.csv', dtype=dtypes_psite)


    # 2. Generate merged dataframes

    #   a) Merged dataframe of env, pphen & psite (all filters w/o taxnames)
    filter1 = environment.merge(pa_pheno, on='Unnamed: 0').merge(pa_site, on='Unnamed: 0')

    #   b) Merged dataframe of env & pphen only (all filters w/o psite & taxnames)
    filter2 = environment.merge(pa_pheno, on='Unnamed: 0')

    # Sanity checks:
    #print(f'Env file: {environment.shape}\tPphen file: {pa_pheno.shape}\tPsite file: {pa_site.shape}\tFilter1: '
    #      f'{filter1.shape}\tFilter2: {filter2.shape}')
    #print(f'{environment.head()}\t{pa_pheno.head()}\t{pa_site.head()}')
    #print(f'{filter1.head()}\t{filter2.head()}')


    # 3. Generate new csv files from the merged dataframes
    #   a) All filters w/o taxnames
    filter1.to_csv('./data/env_pphen_psite_merged.csv', sep=',', encoding='utf-8', index=False)

    #   b) all filters w/o psite & taxnames
    filter2.to_csv('./data/env_pphen_merged.csv', sep=',', encoding='utf-8', index=False)

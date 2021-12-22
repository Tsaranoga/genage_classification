""" Gwendolyn Gusak, Tobias Woertwein """

import pandas as pd

if __name__ == '__main__':

    # 1. Import the csv files
    #   a) abundance file
    col_names = pd.read_csv('./data/PGPT_abundance.csv', sep=';', nrows=0).columns
    # genes = col_names[1:]
    dtypes_abund = {'column=genes - rows=strains': str}
    dtypes_abund.update({col: int for col in col_names if col not in dtypes_abund})

    abundance = pd.read_csv('./data/PGPT_abundance.csv', sep=';', dtype=dtypes_abund)

    strains = abundance['column=genes - rows=strains'].tolist()

    #    b) taxnames file
    taxnames = pd.read_csv('./data/taxnames.csv', sep=';', dtype=str)

    # separate strains according to unclassified phyla
    unclassified_taxa = taxnames[taxnames['PHYLUM'] == 'unclassified']
    classified_taxa = taxnames[taxnames['PHYLUM'] != 'unclassified']
    classified_strains = classified_taxa['IMG GENOME ID'].tolist()

    # Write classified taxa in file for further use
    classified_taxa.to_csv('./data/classified_taxa.csv', sep=',', encoding='utf-8', index=False)

    # subset abundances such that only strains with classified phyla persist
    filtered_abundance = abundance.loc[abundance['column=genes - rows=strains'].isin(classified_strains)]

    #    c) Rest of the files
    # with merged env, pphen & psite
    col_names1 = pd.read_csv('./data/env_pphen_psite_merged.csv', nrows=0).columns
    dtypes1 = {'Unnamed: 0': str}
    dtypes1.update({col: int for col in col_names1 if col not in dtypes1})

    env_pphen_psite = pd.read_csv('./data/env_pphen_psite_merged.csv', dtype=dtypes1)
    env_pphen_psite = env_pphen_psite.loc[env_pphen_psite['Unnamed: 0'].isin(classified_strains)]
    # Write to file version filtered for classified phyla only
    env_pphen_psite.to_csv('./data/env_pphen_psite_merged_filtered.csv', sep=',', encoding='utf-8', index=False)

    # subset filtered abundances such that only strains with env, pphen & psite exist
    env_pphen_psite_strains = env_pphen_psite['Unnamed: 0'].tolist()
    filtered_abundance1 = abundance.loc[abundance['column=genes - rows=strains'].isin(env_pphen_psite_strains)]
    filtered_abundance1.to_csv('./data/fully_filtered_abundances_3files.csv', sep=',', encoding='utf-8', index=False)

    # with merged env, pphen
    col_names2 = pd.read_csv('./data/env_pphen_merged.csv', nrows=0).columns
    dtypes2 = {'Unnamed: 0': str}
    dtypes2.update({col: int for col in col_names2 if col not in dtypes1})

    env_pphen = pd.read_csv('./data/env_pphen_merged.csv', dtype=dtypes2)
    env_pphen = env_pphen.loc[env_pphen['Unnamed: 0'].isin(classified_strains)]
    # Write to file version filtered for classified phyla only
    env_pphen.to_csv('./data/env_pphen_merged_filtered.csv', sep=',', encoding='utf-8', index=False)

    # subset filtered abundances such that only strains with env & pphen exist
    env_pphen_strains = env_pphen['Unnamed: 0'].tolist()
    filtered_abundance2 = abundance.loc[abundance['column=genes - rows=strains'].isin(env_pphen_strains)]
    filtered_abundance2.to_csv('./data/fully_filtered_abundances_2files.csv', sep=',', encoding='utf-8', index=False)

    # Sanity Checks:
    #print(f'Original: {abundance.shape}')
    #print(f'{abundance.head()}\n')

    #print(f' W/o unclassified: {filtered_abundance.shape}')
    #print(f'{filtered_abundance.head()}\n')

    #print(f' Only with env, pphen & psite: {filtered_abundance1.shape}')
    #print(f'{filtered_abundance1.head()}\n')

    #print(f' Only with env & pphen: {filtered_abundance2.shape}')
    #print(f'{filtered_abundance2.head()}\n')

    #print(env_pphen_strains)
    #print(env_pphen_psite_strains)

    #print(taxnames.ndim)
    #print(taxnames.dtypes)
    #print(taxnames.shape)
    #print(taxnames.head())
    # Check for duplicates: print(taxnames.duplicated().sum())
    #print(classified_taxa.shape)
    #print(classified_taxa.head())
    #print(abundance['column=genes - rows=strains'])
    #print(classified_taxa['IMG GENOME ID'])

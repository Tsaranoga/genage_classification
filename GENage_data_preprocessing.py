"""
    Generates files filtered_abundance.csv, filtered_environment.csv, filtered_pa_pheno.csv
    and filtered_pa_site.csv for multi-output
"""

import pandas as pd
from typing import Tuple

__author__ = "Gwendolyn Gusak, Tobias Woertwein"


def read_csv_file(filepath: str, separator: bool, data_types: Tuple[str, str]) -> pd.DataFrame:
    """

    :param filepath: String specifying the file path for input file
    :param separator: True = ',' & False = ';'
    :param data_types: Tuple specifying the data types for first column & rest of the columns
    :return: Pandas dataframe generated from input file
    """

    if separator:
        s = ','
    else:
        s = ';'

    col_names = pd.read_csv(filepath, sep=s, nrows=0).columns
    first_col_name = col_names[0]
    dtypes = {first_col_name: data_types[0]}
    dtypes.update({col: data_types[1] for col in col_names if col not in dtypes})

    dataframe = pd.read_csv(filepath, sep=s, dtype=dtypes)

    return dataframe


def drop_cols(dataframe: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """

    :param dataframe: Input pandas dataframe
    :param threshold: Integer specifying the minimum amount of entries required for each column in input dataframe
    :return: Pandas dataframe containing all columns that meet at least the threshold
    """

    cols_to_drop = []

    for i in dataframe.columns:
        if int(dataframe.sum(axis=0).loc[i]) < threshold:
            cols_to_drop.append(i)

    dataframe = dataframe.drop(cols_to_drop, axis=1)

    return dataframe


if __name__ == '__main__':

    ''' Import all csv files '''
    #   a) abundance file
    abundance = read_csv_file('./data/PGPT_abundance.csv', False, ('str', 'int'))
    abundance = drop_cols(abundance, 1)
    abundance.name = 'abundance'

    #   b) env file
    environment = read_csv_file('./data/env.csv', True, ('str', 'int'))
    environment.name = 'environment'

    #   c) pphen file
    pa_pheno = read_csv_file('./data/pphen.csv', True, ('str', 'int'))
    pa_pheno.name = 'pa_pheno'

    #   d) psite file
    pa_site = read_csv_file('./data/psite.csv', True, ('str', 'int'))
    pa_site.name = 'pa_site'

    #   e) taxnames file
    taxnames = read_csv_file('./data/taxnames.csv', False, ('str', 'str'))

    ''' Filter all files such that only strains remain where PHYLUM is classified '''
    # Separate strains according to unclassified phyla
    unclassified_taxa = taxnames[taxnames['PHYLUM'] == 'unclassified']
    classified_taxa = taxnames[taxnames['PHYLUM'] != 'unclassified']
    classified_strains = classified_taxa['IMG GENOME ID'].tolist()

    # Write classified taxa in file for further use
    classified_taxa.to_csv('./data/classified_taxa.csv', sep=',', encoding='utf-8', index=False)

    # Subset all other files such that only strains with classified phyla persist
    # & Save the filtered dataframes to files

    df_list = [abundance, environment, pa_pheno, pa_site]

    for df in df_list:
        file = f'./data/filtered_{df.name}.csv'
        tmp_df = df.loc[df[df.columns[0]].isin(classified_strains)]

        # print(f'Shape filtered {df.name}: {tmp_df.shape}')
        tmp_df.to_csv(file, sep=',', encoding='utf-8', index=False)

    ''' Sanity Checks '''
    #print(f'Abundance: {abundance.shape}')
    #print(f'{abundance.head()}\n')

    #print(taxnames.ndim)
    #print(taxnames.dtypes)
    #print(taxnames.shape)
    #print(taxnames.head())
    #Check for duplicates: print(taxnames.duplicated().sum())

    #print(f'Env file\tPphen file\tPsite file\n{environment.shape}\t{pa_pheno.shape}\t{pa_site.shape}')
    #print(f'{environment.head()}\t{pa_pheno.head()}\t{pa_site.head()}')

    #print(f' Only with env, pphen & psite: {filtered_abundance1.shape}')
    #print(f'{filtered_abundance1.head()}\n')

    #print(f' Only with env & pphen: {filtered_abundance2.shape}')
    #print(f'{filtered_abundance2.head()}\n')

    #print(env_pphen_strains)
    #print(env_pphen_psite_strains)

    #print(classified_taxa.shape)
    #print(classified_taxa.head())
    #print(abundance['column=genes - rows=strains'])
    #print(classified_taxa['IMG GENOME ID'])

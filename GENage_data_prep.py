""" Gwendolyn Gusak, Tobias Woertwein """

import pandas as pd

# Import the csv files
# abundance file
col_names = pd.read_csv('./data/PGPT_abundance.csv', sep=';', nrows=0).columns
# genes = col_names[1:]
dtypes = {'column=genes - rows=strains': str}
dtypes.update({col: int for col in col_names if col not in dtypes})

abundance = pd.read_csv('./data/PGPT_abundance.csv', sep=';', dtype=dtypes)

strains = abundance['column=genes - rows=strains'].tolist()

# taxnames file
taxnames = pd.read_csv('./data/taxnames.csv', sep=';', dtype=str)

# separate strains according to unclassified phyla
unclassified_taxa = taxnames[taxnames['PHYLUM'] == 'unclassified']
classified_taxa = taxnames[taxnames['PHYLUM'] != 'unclassified']

# subset abundances such that only strains with classified phyla persist
filtered_abundance = abundance[abundance.set_index('column=genes - rows=strains').index.isin(classified_taxa.set_index('IMG GENOME ID').index)]


if __name__ == '__main__':

    #print(taxnames.ndim)
    #print(taxnames.dtypes)
    #print(taxnames.shape)
    #print(taxnames.head())
    # Check for duplicates: print(taxnames.duplicated().sum())
    #print(classified_taxa.shape)
    #print(classified_taxa.head())
    #print(abundance['column=genes - rows=strains'])
    #print(classified_taxa['IMG GENOME ID'])

    print(abundance.shape)
    print(filtered_abundance.shape)

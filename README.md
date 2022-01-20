# GENageClassPGPT
***
GENageCLassPGPT is a machine learning based image classifier. 
This tool classifies images of gene occurrence or abundance matrices (genages) into different classes related to isolation site (env), plant sphere (psite) and 
phenotype (pphen). Where phenotype specifies plant-associated pathogens or 
plant-associated symbionts.

## Data preprocessing
To filter the data GENage_data_preprocessing.py needs to be applied. This 
program reads in all files as pandas dataframe, renders it such that strains 
that are unclassified in phylum and are not contained in any strain are removed.
The filtered files are saved to the data folder in the current directory.

__Disclaimer:__ This program is currently hard-coded on the data from the project.

## Input generation
To generate input and ground truth dictionaries along with the genages GENage_data.py is applied to the filtered files. While parsing the abundance 
file a gene list and the input dictionary is generated. To get matrices with the 
abundance values comment line 61 out otherwise the matrices contain occurrence values. For the abundance values the usage of the function __dictionary_to_images__ should be commented out as well as this function is 
only fitted to the occurrence values.  With this function images for the occurrence values are calculated from the matrices contained in the input dictionary. While this program parses the ground truth files (env, psite, pphen) classes contained in these files are filtered such that only classes remain that contain at least 50 entries. The file taxnames is also parsed and transferred into a dictionary. All lists and dictionaries are saved as pickle files in "./data/pickle_files".  

__Disclaimer:__ Genages generated with __dictionary_to_images__ are currently 
not saved in the directory but on another hard drive. Before running the code please change this.

__Disclaimer:__ This program is currently hard-coded on the data from the project.

__Hint:__ For the main program GENageClassPGPT.py lines 61 and 247 are obsolete 
and should be commented out.

## Network: GENageClassPGPT

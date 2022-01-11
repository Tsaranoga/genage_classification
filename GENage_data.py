"""
    Generates pickle files for all dictionaries for input and all ground truth files ALONG WITH
    the pickle files for all lists containing strain IDs for training, validation and test subsetting
    splitted according to equal ratio and obtained for each ground truth file. Additionally a pickle
    file containing a list of gene names is generated.
    Required as input: Generated files from program GENage_data_preprocessing.py
"""

import csv
import cv2
import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Union, Tuple

__author__ = "Gwendolyn Gusak, Tobias Woertwein"


def parse_dict_from_file(filepath: str, delim: str = ',') -> Union[dict, Tuple[list, dict]]:
    """

    :param filepath: String specifying the input file path
    :param delim: String specifying the delimiter of the file
    :return: Dictionary either for input or one of the outputs
    """

    dictionary = {}
    class_dict = {}
    genes = []

    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter=delim)

        for index, row in enumerate(reader):

            # matrix = np.zeros((80, 80), dtype=int)

            if index == 0:
                header_row = row

                if len(header_row) == 6356:

                    for gene in header_row[1:]:
                        genes.append(gene)

            else:
                if index != 0:

                    current_row = row
                    strain = current_row[0]

                    if len(header_row) == 6356:

                        matrix_as_list = [int(x) for x in current_row[1:]]
                        # print(len(matrix_as_list))
                        # matrix[:] = matrix_as_list
                        matrix = np.array(matrix_as_list, dtype=int)
                        matrix.resize((80, 80))

                        matrix = np.where(matrix == 0, 0, 1)

                        dictionary.update({strain: matrix})

                    elif len(header_row) == 7:

                        traits_dict = {}

                        for i in range(1, len(current_row)):

                            traits_dict.update({header_row[i]: int(current_row[i])})

                        dictionary.update({strain: traits_dict})

                    elif len(header_row) == 8 and header_row[0] == 'IMG GENOME ID':

                        taxnames_dict = {}

                        for j in range(1, len(current_row)):
                            taxnames_dict.update({header_row[j]: current_row[j]})

                        dictionary.update({strain: taxnames_dict})

                    else:

                        file_dict = {}
                        current_class = ''

                        for k in range(1, len(current_row)):
                            file_dict.update({header_row[k]: int(current_row[k])})
                            current_class += current_row[k]

                        if current_class in class_dict.keys():
                            class_dict.get(current_class, []).append(strain)
                        else:
                            class_dict.update({current_class: [strain]})

                        dictionary.update({strain: file_dict})

    if class_dict:

        tmp_dict = {}

        for class_strains in class_dict.values():

            if len(class_strains) >= 50:

                for strain in class_strains:
                    tmp_dict.update({strain: dictionary.get(strain)})

        dictionary = tmp_dict

    if genes:

        return genes, dictionary

    return dictionary


def dictionary_to_images(input_dict: dict):
    """
    Turns matrices from input dictionary in images with key as image name
    :param input_dict: Dictionary containing matrices as values
    """

    for strain, matrix in input_dict.items():
        filename = f'G:/Input_images_GENage/{strain}.png'
        matrix = np.where(matrix == 0, 0, 255)
        cv2.imwrite(filename, matrix)


'''
def generate_bin_dict(gt_traits: dict, traits_oi: list):

    bin_dict = {}

    for strain in gt_traits.keys():
        current_bin = []
        traits_dict = gt_traits.get(strain)

        for trait in traits_oi:
            current_trait = traits_dict.get(trait)
            current_bin.append(current_trait)

        bin_dict.update({strain: current_bin})

    cls = np.array([str(i) for i in bin_dict.values()])
    print(np.unique(cls, return_counts=True))  # Remove strains with class < 10/50

    return bin_dict


def generate_train_validation_test_splits(data: dict, traits_oi: list):

    # 1. Preparation for first split into training and temporary indices lists
    bin_dict = generate_bin_dict(data, traits_oi)  # get dictionary with trait bins
    key_list = list(bin_dict)  # extract strains from bin_dict

    training = []
    tmp = []

    x = np.zeros(len(key_list))  # original amount of strains
    y = np.array(list(bin_dict.values()))  # original bins derived by traits per strain

    print(f'Before first split: x={len(key_list)}, y={y.sum(axis=0)}')

    # 2. First split of strains into training & temporary indices lists
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=7)
    train_indices, tmp_indices = next(splitter1.split(x, y))

    #   a) Extract list of training strains
    for train_i in train_indices:
        training.append(key_list[train_i])

    #   b) Extract list of remaining/ temporary strains
    for tmp_i in tmp_indices:
        tmp.append(key_list[tmp_i])

    # 3. Preparation for second split into validation and test indices lists
    new_traits_dict = {key: data[key] for key in tmp}  # extract traits for remaining strains
    new_bin_dict = generate_bin_dict(new_traits_dict, traits_oi)  # get dictionary with trait bins
    new_key_list = list(new_bin_dict)  # extract strains from new_bin_dict

    validation = []
    test = []

    new_x = np.zeros(len(new_key_list))  # amount of remaining strains
    new_y = np.array(list(new_bin_dict.values()))  # new bins derived by traits per remaining strain

    print(f'Before second split: x={len(new_key_list)}, y={new_y.sum(axis=0)}')

    # 4. Second split of strains into validation & test indices lists
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=7)
    validation_indices, test_indices = next(splitter2.split(new_x, new_y))

    #   a) Extract list of validation strains
    for validation_i in validation_indices:
        validation.append(new_key_list[validation_i])

    #   b) Extract list of test strains
    for test_i in test_indices:
        test.append(new_key_list[test_i])

    n_traits_dict = {key: data[key] for key in test}  # extract traits for remaining strains
    n_bin_dict = generate_bin_dict(n_traits_dict, traits_oi)  # get dictionary with trait bins
    n_key_list = list(n_bin_dict)  # extract strains from new_bin_dict

    n_x = np.zeros(len(n_key_list))  # amount of remaining strains
    n_y = np.array(list(n_bin_dict.values()))  # new bins derived by traits per remaining strain

    print(f'After second split: x={len(n_key_list)}, y={n_y.sum(axis=0)}')

    # 5. Return relevant lists containing strains for the training, validation or test, respectively
    return training, validation, test
'''


def generate_bin_dict(gt_dict: dict) -> dict:
    """

    :param gt_dict: Input dictionary
    :return: Dictionary containing according class to key
    """

    bin_dict = {}

    for strain in gt_dict.keys():
        current_bin = []
        categories_dict = gt_dict.get(strain)

        for category in categories_dict:
            current_category = categories_dict.get(category)
            current_bin.append(current_category)

        bin_dict.update({strain: current_bin})

    #cls = np.array([str(i) for i in bin_dict.values()])
    #print(np.unique(cls, return_counts=True))  # Remove strains with class < 10/50

    return bin_dict


def generate_train_validation_test_splits(data: dict) -> (list, list, list):
    """

    :param data: An output/ ground truth dictionary
    :return: Lists of the keys from data for training, validation and test subsetting
    """

    # 1. Preparation for first split into training and temporary indices lists
    bin_dict = generate_bin_dict(data)  # get dictionary with trait bins
    key_list = list(bin_dict)  # extract strains from bin_dict

    training = []
    tmp = []

    x = np.zeros(len(key_list))  # original amount of strains
    y = np.array(list(bin_dict.values()))  # original bins derived by traits per strain

    #print(f'Before first split: x={len(key_list)}, y={y.sum(axis=0)}')

    # 2. First split of strains into training & temporary indices lists
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=7)
    train_indices, tmp_indices = next(splitter1.split(x, y))

    #   a) Extract list of training strains
    for train_i in train_indices:
        training.append(key_list[train_i])

    #   b) Extract list of remaining/ temporary strains
    for tmp_i in tmp_indices:
        tmp.append(key_list[tmp_i])

    # 3. Preparation for second split into validation and test indices lists
    new_traits_dict = {key: data[key] for key in tmp}  # extract traits for remaining strains
    new_bin_dict = generate_bin_dict(new_traits_dict)  # get dictionary with trait bins
    new_key_list = list(new_bin_dict)  # extract strains from new_bin_dict

    validation = []
    test = []

    new_x = np.zeros(len(new_key_list))  # amount of remaining strains
    new_y = np.array(list(new_bin_dict.values()))  # new bins derived by traits per remaining strain

    #print(f'Before second split: x={len(new_key_list)}, y={new_y.sum(axis=0)}')

    # 4. Second split of strains into validation & test indices lists
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=7)
    validation_indices, test_indices = next(splitter2.split(new_x, new_y))

    #   a) Extract list of validation strains
    for validation_i in validation_indices:
        validation.append(new_key_list[validation_i])

    #   b) Extract list of test strains
    for test_i in test_indices:
        test.append(new_key_list[test_i])

    '''
    n_traits_dict = {key: data[key] for key in test}  # extract traits for remaining strains
    n_bin_dict = generate_bin_dict(n_traits_dict)  # get dictionary with trait bins
    n_key_list = list(n_bin_dict)  # extract strains from new_bin_dict

    n_x = np.zeros(len(n_key_list))  # amount of remaining strains
    n_y = np.array(list(n_bin_dict.values()))  # new bins derived by traits per remaining strain

    print(f'After second split: x={len(n_key_list)}, y={n_y.sum(axis=0)}')
    '''

    # 5. Return relevant lists containing strains for the training, validation or test, respectively
    return training, validation, test

'''
def split_input_data(gt_traits: dict):

    env_train, env_valid, env_test = generate_train_validation_test_splits(gt_traits, ['PA', 'SA'])
    pphen_train, pphen_valid, pphen_test = generate_train_validation_test_splits(gt_traits, ['PA_SYMB', 'PA_PATH'])
    psite_train, psite_valid, psite_test = generate_train_validation_test_splits(gt_traits, ['PA_PHYL', 'PA_RHIZ'])

    return env_train, env_valid, env_test, pphen_train, pphen_valid, pphen_test, psite_train, psite_valid, psite_test
'''


#  Store dictionaries and lists in pickle file
def store_as_pickle(data: Union[dict, list], filename: str):
    """

    :param data: Either dictionary or list
    :param filename: String specifying the name for the pickle file
    """

    filepath = f'./data/pickle_files/{filename}.p'

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':

    ''' Generate input '''
    # Input dictionary
    # input_dict_so = parse_dict_from_file('./data/filtered_abundance_single_output.csv') (only for single output)
    gene_list, input_dict = parse_dict_from_file('./data/filtered_abundance.csv')
    dictionary_to_images(input_dict)
    #   Store input dictionary in pickle file
    store_as_pickle(input_dict, 'input')
    store_as_pickle(gene_list, 'gene_list')

    ''' Generate ground truth (gt) dictionaries '''

    ''' Required for single output
    #   a1) Traits gt dictionary for single output
    gt_traits = parse_dict_from_file('./data/single_output.csv')
    #env_train, env_valid, env_test, pphen_train, pphen_valid, pphen_test, psite_train, psite_valid, psite_test =\
    #    split_input_data(gt_traits)

    #   Store gt_traits in pickle file
    store_as_pickle(gt_traits, 'gt_traits')
    '''

    #   a2) Single dictionaries for environment, pa_pheno & pa_site
    #       I) Environment gt dictionary
    gt_environment = parse_dict_from_file('./data/filtered_environment.csv')
    store_as_pickle(gt_environment, 'gt_environment')
    env_training, env_validation, env_test = generate_train_validation_test_splits(gt_environment)
    store_as_pickle(env_training, 'env_training')
    store_as_pickle(env_validation, 'env_validation')
    store_as_pickle(env_test, 'env_test')

    #       II) Pa_pheno gt dictionary
    gt_pa_pheno = parse_dict_from_file('./data/filtered_pa_pheno.csv')
    store_as_pickle(gt_pa_pheno, 'gt_pa_pheno')
    pphen_training, pphen_validation, pphen_test = generate_train_validation_test_splits(gt_pa_pheno)
    store_as_pickle(pphen_training, 'pphen_training')
    store_as_pickle(pphen_validation, 'pphen_validation')
    store_as_pickle(pphen_test, 'pphen_test')

    #       III) Pa_site gt dictionary
    gt_pa_site = parse_dict_from_file('./data/filtered_pa_site.csv')
    store_as_pickle(gt_pa_site, 'gt_pa_site')
    psite_training, psite_validation, psite_test = generate_train_validation_test_splits(gt_pa_site)
    store_as_pickle(psite_training, 'psite_training')
    store_as_pickle(psite_validation, 'psite_validation')
    store_as_pickle(psite_test, 'psite_test')

    #   b) Taxnames gt dictionary
    gt_taxnames = parse_dict_from_file('./data/classified_taxa.csv')

    #   Store input dictionary in pickle file
    store_as_pickle(gt_taxnames, 'gt_taxnames')

    ''' Sanity checks '''
    #input_items = input_dict.items()
    #print(f'Input: {list(input_items)[:1]}')
    #np.set_printoptions(threshold=np.inf)
    #print(input_dict.get('2228664006'))
    #input_items_so = input_dict_so.items()
    #print(f'Input for so: {list(input_items_so)[:1]}')

    #gt_environment_items = gt_environment.items()
    #print(f'Environment: {list(gt_environment_items)[:1]}')

    #gt_pa_pheno_items = gt_pa_pheno.items()
    #print(f'Pa_pheno: {list(gt_pa_pheno_items)[:1]}')

    #gt_pa_site_items = gt_pa_site.items()
    #print(f'Pa_site: {list(gt_pa_site_items)[:1]}')
    
    #gt_traits_items = gt_traits.items()
    #print(f'Traits: {list(gt_traits_items)[:1]}')

    #gt_taxnames_items = gt_taxnames.items()
    #print(f'Taxnames: {list(gt_taxnames_items)[:1]}')

""" Gwendolyn Gusak, Tobias Woertwein """

import csv
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def parse_dict_from_file(filepath: str, delim: str = ','):

    dict = {}

    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter=delim)

        for index, row in enumerate(reader):

            # matrix = np.zeros((80, 80), dtype=int)

            if index == 0:
                header_row = row

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

                        dict.update({strain: matrix})

                    elif len(header_row) == 10 or len(header_row) == 15:

                        traits_dict = {}

                        for i in range(1, len(current_row)):
                            traits_dict.update({header_row[i]: int(current_row[i])})

                        dict.update({strain: traits_dict})

                    elif len(header_row) == 8:

                        taxnames_dict = {}

                        for i in range(1, len(current_row)):
                            taxnames_dict.update({header_row[i]: current_row[i]})

                        dict.update({strain: taxnames_dict})

    return dict


def dictionary_to_images(input_dict: dict):

    for strain, matrix in input_dict.items():
        filename = f'G:/Input_images_GENage/{strain}.png'
        matrix = np.where(matrix == 0, 0, 255)
        cv2.imwrite(filename, matrix)


def split_input_data(gt_traits: dict):

    bin_dict = {}

    for strain in gt_traits.keys():
        current_bin = ''
        traits_dict = gt_traits.get(strain)

        for trait in traits_dict.keys():
            current_trait = str(traits_dict.get(trait))
            current_bin += current_trait

        bin_dict.update({strain: current_bin})

    x = np.zeros(len(gt_traits.keys()))
    y = np.array(list(bin_dict.values()))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=3)
    for train_index, test_index in splitter.split(x, y):
        print(f'Train indices: {train_index}\tTest indices: {test_index}')



if __name__ == '__main__':

    # Input dictionary
    input_dict = parse_dict_from_file('./data/fully_filtered_abundances_2files.csv')
    # input_dict = parse_dict_from_file('./data/fully_filtered_abundances_3files.csv')
    dictionary_to_images(input_dict)


    # Ground truth (gt) dictionnaries
    #   a) Traits gt dictionary
    gt_traits = parse_dict_from_file('./data/env_pphen_merged_filtered.csv')
    # gt_traits = parse_dict_from_file('./data/env_pphen_psite_merged_filtered.csv')
    # training, evaluation, test =
    split_input_data(gt_traits)

    #   b) Taxnames gt dictionary
    gt_taxnames = parse_dict_from_file('./data/classified_taxa.csv')


    # Sanity checks
    '''
    input_items = input_dict.items()
    print(f'Input: {list(input_items)[:1]}')
    np.set_printoptions(threshold=np.inf)
    print(input_dict.get('2228664006'))
    
    gt_traits_items = gt_traits.items()
    print(f'Traits: {list(gt_traits_items)[:1]}')

    gt_taxnames_items = gt_taxnames.items()
    print(f'Taxnames: {list(gt_taxnames_items)[:1]}')
    '''

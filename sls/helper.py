import numpy as np


def search(dictionary, value):
    for key in dictionary.keys():
        if np.array_equal(value, dictionary[key]):
            return key


def group_into_states(elements_1, elements_2):

    states = [(2 * (elements_1 - - 64) / (64 - -64) ) - 1, (2 * (elements_2 - -64) / (64 - -64) ) - 1]
    return states


def create_states():
    coordinates = list(range(-64, 65))
    groups = []
    for el in coordinates:
        for el2 in coordinates:
            group = group_into_states(el, el2)
            groups.append(group)
    print(len(groups))
    return groups


def convert_array_to_string(int_list):
    return " ".join(str(x) for x in int_list)

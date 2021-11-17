def search(dictionary, value):
    value = list(value)
    for key in dictionary.keys():
        if value in dictionary[key]:
            return key


def group_into_states(elements_1, elements_2):
    states = []
    for x in elements_1:
        for y in elements_2:
            states.append([x, y])
    if [0, 0] in states:
        states.remove([0, 0])
    print(states)
    return states


def create_states():
    coordinates = list(range(-64, 64))
    grouped_coordinates = []

    while True:
        split_numbers = []
        if len(coordinates) == 0:
            break
        for i in range(4):      # TODO: Try range 4 and 2
            split_numbers.append(coordinates.pop(0))

        grouped_coordinates.append(split_numbers)

    groups = []
    index = 0

    for el in grouped_coordinates:
        for el2 in grouped_coordinates:
            index += 1
            group = group_into_states(el, el2)
            groups.append((index, group))
    groups.append((0,[[0, 0]]))
    states_list = dict(groups)
    return states_list, states_list.keys()


def convert_array_to_string(int_list):
    return " ".join(str(x) for x in int_list)

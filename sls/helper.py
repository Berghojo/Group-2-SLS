def search(dict, value):
    value = list(value)
    for key in dict.keys():
        if value in dict[key]:
            return key

def group(el1, el2):
    grouped_states = []
    for i in el1:
        for e in el2:
            grouped_states.append([i, e])
    return grouped_states


def create_states():
    states = []
    #number = [-1, 0, 1]
    number = list(range(-64,64))
    split_array = []

    while True:
        split_number = []
        if len(number) == 0:
            break
        for i in range(4):
            split_number.append(number.pop(0))

        split_array.append(split_number)


    groups = []
    i = 0
    for el in split_array:
        for el2 in split_array:
            i += 1
            grup = group(el, el2)
            groups.append((i, grup))
    states = dict(groups)
    return states, states.keys()
states, index = create_states()
print(search(states, [34, 31]))

'''
    for i in number:
        for e in number:
            states.append(" ".join(str(x) for x in [i, e]))
    return states

'''

def convert_array_to_string(int_list):
    return " ".join(str(x) for x in int_list)

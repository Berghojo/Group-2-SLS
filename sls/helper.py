def create_states():
    states = []
    number = [-1, 0, 1]
    for i in number:
        for e in number:
            states.append(" ".join(str(x) for x in [i, e]))
    return states


def convert_array_to_string(int_list):
    return " ".join(str(x) for x in int_list)

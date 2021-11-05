def create_states(number):
    states = []
    for i in range(-number, number):
        for e in range(-number, number):
            states.append(" ".join(str(x) for x in [i, e]))
    return states


def convert_array_to_string(int_list):
    return " ".join(str(x) for x in int_list)

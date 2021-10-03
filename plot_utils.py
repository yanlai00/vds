def max_length_in_list_of_list(lst):
    max_length = 0
    for i in range(len(lst)):
        if len(lst[i]) > max_length:
            max_length = len(lst[i])
    return max_length

def min_length_in_list_of_list(lst):
    min_length = float('inf')
    for i in range(len(lst)):
        if len(lst[i]) < min_length:
            min_length = len(lst[i])
    return min_length

def truncate_list_of_list_to_rectangular(lst):
    min_length = min_length_in_list_of_list(lst)
    for i in range(len(lst)):
        lst[i] =  lst[i][:min_length]
    return lst, min_length
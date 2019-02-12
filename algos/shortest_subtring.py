def find_length_greedy(input_string, char_set):
    """
    Greedy O(n^2)
    """
    all_substrings = []
    for start_idx in range(len(input_string)):
        for end_idx in range(len(input_string), start_idx, -1):
            all_substrings.append(input_string[start_idx:end_idx])
    shortest_len = float("inf")
    for i in all_substrings:
        if all_chars_in_str(i, char_set) and len(i) < shortest_len:
            shortest_len = len(i)
    return shortest_len

def all_chars_in_str(input_string, char_set):
    for i in char_set:
        if i not in input_string:
            return False
    return True

def find_length(input_string, char_set):
    """
    Greedy O(n^2)
    """

input_string = "abdbfxxdaf"
char_set = ["a", "d", "f"]
shortest_len = find_length_greedy(input_string, char_set)
assert 3 == shortest_len # "daf"
input_string = "this is a test string"
char_set = ["t", "i"]
shortest_len = find_length_greedy(input_string, char_set)
assert 3 == shortest_len # "t stri"
my_dict = {'a': 0, 'b': 1, 'c':2}

try:
    my_dict['d']
except KeyError:
    print('KeyError!!')

from collections import defaultdict

def default_val():
    return None

my_default_dict = defaultdict(default_val)
my_default_dict['a'] = 0
my_default_dict['b'] = 1
my_default_dict['c'] = 2

print(my_default_dict['d'])

my_default_dict_2 = defaultdict(lambda: None)
my_default_dict_2['a'] = 0
my_default_dict_2['b'] = 1
my_default_dict_2['c'] = 2
print(my_default_dict_2['d'])
import os
import numpy as np

pubfig83_test_path = 'recognizer/pubfig83-test'

all_people_names = os.listdir(pubfig83_test_path)
all_people_names = np.sort(all_people_names)

all_people_class_nums = np.arange(83)

dictionary = dict(zip(all_people_names, all_people_class_nums))

np.save('people-class_num-dictionary.npy', dictionary)


read_dictionary = np.load('people-class_num-dictionary.npy', allow_pickle = 'TRUE').item()
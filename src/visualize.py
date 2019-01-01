import os
import pickle
import pandas as pd

# root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_dir_path = '/Users/davidwallach/Desktop/slack_bot/alfred'
X_TEST_PATH = root_dir_path + '/data/X_test'
Y_TEST_PATH = root_dir_path + '/data/Y_test'

file = open(X_TEST_PATH, 'rb')
X_test = pickle.load(file)[:20]
file.close()

# file = open(Y_TEST_PATH, 'rb')
# Y_test = pickle.load(file)
# file.close()

X_test.to_csv(root_dir_path + '/data/X_test_small.csv')
# Y_test.to_csv(root_dir_path + '/data/Y_test.csv')

X_test

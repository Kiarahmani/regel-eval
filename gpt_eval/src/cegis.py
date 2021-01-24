import pandas as pd
import re
import time
import ast
import subprocess
import glob
import os
import math
import shlex
from subprocess import Popen, PIPE
from threading import Timer
from chardet import detect
import numpy as np
import logging
from pprint import pprint
import traceback
import json
import collections
from enum import Enum
import yaml
import util
import constants

# set up files
dirname = os.path.dirname(__file__)
conf_file = os.path.join(dirname, './config.yaml')
conf = yaml.safe_load(open(conf_file))

dataset_file = os.path.join("dataset", conf['dataset']['file_name'])
matches_file = os.path.join("cache", conf['cache']['match_file'])
example_file = os.path.join("cache", conf['cache']['example_file'])

# initialize logger
logging.basicConfig(format='[%(levelname)s] [%(asctime)s] [%(funcName)-25s%(lineno)s]  %(message)s', level=logging.DEBUG)

# range of included rows from the dataset
included_rows = range(conf['dataset']['min_range'],conf['dataset']['max_range'])



def eval_gpt_prune_candidates():
    logging.debug("begin analysis")
    dt = pd.read_csv(dataset_file, encoding=util.get_encoding_type(dataset_file))
    results = {str(i):[] for i in range(conf['user_interactions_cap'] + 1)}
    for row in dt.iterrows():
        # make sure current row is included for analysis
        if (not row[0] in included_rows) or (row[0] in constants.BAD_ROWS):
            logging.debug("row " + str(row[0]) + " is not in included range. skipping")
            continue
        expected = row[1]['expected output'].replace(".","(.)")
        # make sure current row's expected regex does not include any of the bad characters
        if len([i for i in constants.BAD_CHARS if i in expected]) > 0:
            logging.debug("row " + str(row[0]) + " has a bad character, skipping")
            continue
        logging.debug("row " + str(row[0]) + " is accepted for analysis.\n\n" + "*" * 75 + "\n" + str(row[1]) + "\n" + "*" * 75 + "\n")
        candidates = []
        for i in range(0,conf['dataset']['cand_count']):
            candidate = row[1]['candidate' + str(i)]
            if (type(candidate) == str) or (not math.isnan(candidate)):
                candidates.append(candidate.replace(".","(.)"))
        logging.debug("beginning analysis for the given list of candidates: "+str(candidates))
        found = False
        empty = False
        for i in range(conf['user_interactions_cap']+1):
            if empty: # all candidates are already considered; there is no need to perform the analysis
                results[str(i)].append(False)
                continue
            if not found:
                logging.debug("loop#"+str(i)+" current candidates: "+str(candidates))
                if len(candidates) == 0:
                    empty = True
                    results[str(i)].append(False)
                    continue
                # check if the first candidate matches the truth or not
                logging.debug("checking the first candidate " + str(candidates[0] + " against the truth"))
                res = util.compare_regex_semantic(expected,candidates[0])
                logging.debug("is equivalent? "+str(res))
                results[str(i)].append(res)
                found = res
                if not found:
                    candidates = candidates[1:]
                #candidates = prune_candidates(row[0], i, expected, candidates)
                candidates = candidates[1:]
                logging.debug("post pruning number of candidates: " + str(len(candidates)))
            else:
                #match is already found; just append True at the end of the results
                results[str(i)].append(True)
    return results




# MAIN
if __name__ == "__main__":    
    start_time = time.time()
    res = eval_gpt_prune_candidates()
    print (res)
    print("# total running time: %s seconds" % (time.time() - start_time))

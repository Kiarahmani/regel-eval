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
import demo


# set up files
dirname = os.path.dirname(__file__)
conf_file = os.path.join(dirname, './config.yaml')
conf = yaml.safe_load(open(conf_file))

dataset_file = os.path.join("dataset", conf['dataset']['file_name'])
matches_file = os.path.join("cache", conf['cache']['match_file'])
example_file = os.path.join("cache", conf['cache']['example_file'])
result_file = os.path.join("results", conf['result']['file_name'])

# initialize logger
logging.basicConfig(format='[%(levelname)s] [%(asctime)s] [%(funcName)-25s%(lineno)s]  %(message)s', level=logging.DEBUG)

# range of included rows from the dataset
included_rows = range(conf['dataset']['min_range'],conf['dataset']['max_range'])



def eval_gpt_prune_candidates():
    dt = pd.read_csv(dataset_file, encoding=util.get_encoding_type(dataset_file))
    results = {str(i):[] for i in range(conf['user_interactions_cap'] + 1)}
    output_rows = []
    for row in dt.iterrows():
        description = row[1]['description']
        expected = row[1]['expected output'].replace(".","(.)")
                # make sure current row is included for analysis and is not broken
        if (not row[0] in included_rows) or (row[0] in constants.BAD_ROWS) or len([i for i in constants.BAD_CHARS if i in expected]) > 0:
            logging.debug("row " + str(row[0]) + " is not in included range. skipping")
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
        top_candidate = None
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
                else:
                    top_candidate = candidates[0]
                # check if the first candidate matches the truth or not
                logging.debug("checking the first candidate " + str(candidates[0] + " against the truth"))
                res = util.compare_regex_semantic(expected,candidates[0])
                logging.debug("is equivalent? "+str(res))
                results[str(i)].append(res)
                found = res
                if not found:
                    candidates = candidates[1:]
                test_case = demo.generate_test_case(candidates)
                candidates = demo.filter_regex_list(candidates, test_case, util.is_match_rfixer(expected, test_case))
                logging.debug("post pruning number of candidates: " + str(len(candidates)))
            else:
                #match is already found; just append True at the end of the results
                results[str(i)].append(True)
        #if not found:
            #output_rows.append([row[0], description, expected, top_candidate])
    #df = pd.DataFrame(output_rows, columns=['org #','description', 'expected output', 'top candidate'])
    #df.to_csv(result_file, encoding='utf-8')
    return results




def eval_repair():
    dt = pd.read_csv(dataset_file, encoding=util.get_encoding_type(dataset_file))
    output_rows = []
    for row in dt.iterrows():
        description = row[1]['description']
        number = row[1]['org #']
        expected = row[1]['expected output'].replace(".","(.)")
        candidate = row[1]['top candidate']
        if (not row[0] in included_rows) or (row[0] in constants.BAD_ROWS) or len([i for i in constants.BAD_CHARS if i in expected]) > 0:
            logging.debug("row " + str(row[0]) + " is not in included range. skipping")
            continue
        final_solution, final_pos, final_neg = util.attempt_repair(candidate, expected)
        pos_len =  len(final_pos) if final_pos != None else 0
        neg_len =  len(final_neg) if final_neg != None else 0
        output_rows.append([number,description, expected, candidate, final_solution, pos_len + neg_len, final_pos, final_neg])

    df = pd.DataFrame(output_rows, columns=['#', 'description','expected output', 'candidate', 'rfixer solution', 'number of examples', 'positive examples', 'negative examples'])
    df.to_csv(result_file, encoding='utf-8')





def eval_repair_without_cegis():
    dt = pd.read_csv(dataset_file, encoding=util.get_encoding_type(dataset_file))
    output_rows = []
    for row in dt.iterrows():
        description = row[1]['description']
        expected = row[1]['expected output'].replace(".","(.)")
        candidate = row[1]['candidate0']
        if (not row[0] in included_rows) or (row[0] in constants.BAD_ROWS) or len([i for i in constants.BAD_CHARS if i in expected]) > 0:
            logging.debug("row " + str(row[0]) + " is not in included range. skipping")
            continue

        candidates = []
        for i in range(0,5):
            candidate = row[1]['candidate' + str(i)]
            if (type(candidate) == str) or (not math.isnan(candidate)):
                candidates.append(candidate.replace(".","(.)"))
        examples = demo.generate_potential_test_case(candidates)
        final_pos = []
        final_neg = []
        for example in examples:
            if util.is_match_rfixer(expected, example):
                final_pos.append(example)
            else:
                final_neg.append(example)

        print ("\n\n")
        print (final_pos)
        print (final_neg)
        print ("\n\n")
        final_solution = demo.repair_regex(candidate, final_pos, final_neg)

        #final_solution, final_pos, final_pos = util.attempt_repair(candidate, expected)
        pos_len =  len(final_pos) if final_pos != None else 0
        neg_len =  len(final_neg) if final_neg != None else 0
        output_rows.append([description, expected, candidate, final_solution, pos_len + neg_len, final_pos, final_neg, util.compare_regex_semantic(expected, final_solution)])

    
    for row in output_rows:
        print (row)
    #df = pd.DataFrame(output_rows, columns=['description','expected output', 'candidate', 'rfixer solution', 'number of examples', 'positive examples', 'negative examples'])
    #df.to_csv(result_file, encoding='utf-8')




# MAIN
if __name__ == "__main__":  
    start_time = time.time()
    eval_repair_without_cegis()
    
    #print (util.attempt_repair("a((.))*((.))*","a(((.))*a)?"))
    #print (util.generate_counter_example("\w","\d"))
    #res = eval_gpt_prune_candidates()
    #print (res)
    #for key, value in res.items():
    #    tr_cnt = len([x for x in value if x is True])
    #    print("success rate with " + key + " examples: " + str((tr_cnt) / len(value) * 100) + "%")
    print("# total running time: %s seconds" % (time.time() - start_time))


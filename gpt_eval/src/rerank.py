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






# given a list of candidates, generates a 2D matrix of strings where cell
# [i][j] is accepted by candidates[i] and is rejected by candidates[j]
def generate_examples_matrix(candidates):
    logging.debug("beginning example matrix generation")
    result = [[] for c in candidates]
    logging.debug("initial empty matrix: " + str(result))
    for i, c1 in enumerate(candidates):
        for j, c2 in enumerate(candidates):
            if i == j:
                result[i].append(None)
                continue
            ce = generate_counter_example(c1,c2)
            logging.debug("generating a counter-example for: " + c1 + " and " + c2)
            logging.debug("counter example: " + str(ce))
            result[i].append(ce)
            logging.debug("current matrix row: " + str(result[i]))
    return result


# returns a dictionary of examples to the list of candidates that accepts them
def identify_devisions(row, candidates, examples_matrix):
    result = dict()
    for l in [e for e in (l1 for l1 in examples_matrix)]:
        for example in l:
            if example is None:
                continue
            #logging.debug("identifying 2 buckets of candidates separated by the given example (look below)")
            #logging.debug("candidates: " + str(candidates))
            #logging.debug("example: " + str(example))
            accepting_cands = []
            for candidate in candidates:
                if is_match(row, candidate, example):
                    accepting_cands.append(candidate)
            if example in result.keys():
                result[example].extend(accepting_cands)
            else:
                result[example] = accepting_cands
            result[example] = list(set(result[example]))
    return result


def is_match_user(example):
    match = input("Is " + example + " a valid match? (y/n) ")
    return (match == 'y')


def is_match_cache(row, regex, example):
    flag = False
    mmap = {}
    with open(MATCH_FILE, 'r', encoding="utf-16") as file:
        for line in file:
            line = line.rstrip()  # remove '\n' at end of line
            if flag:
                mmap = json.loads(line)
                break
            if (line == str(row)):
                flag = True
    if example in mmap.keys():
        return regex in mmap[example]
    else:
        return None


def is_match(row, regex, example):
    res = is_match_cache(row, regex, example)
    if res != None: # cache hit
        return res
    if matcher == Matcher.user:
        return is_match_user(example)
    elif matcher == Matcher.rfixer:
        rfres = is_match_rfixer(row, regex, example)
        return rfres
    elif matcher == Matcher.python:
        return is_match_python(regex, example)
    raise Exception


# returns the previously generated map of examples between regexes
def load_example_matrix(row, candidates):
    logging.debug("loading the example matrix for row "+str(row) +" with candidates "+ str(candidates))
    flag = False
    mmap = {}
    with open(MAP_FILE, 'r') as file:
        for line in file:
            line = line.rstrip()  # remove '\n' at end of line
            if flag:
                mmap = json.loads(line)
                break
            if (line == str(row)):
                flag = True
    result = [[] for c in candidates]
    for i, c1 in enumerate(candidates):
        for j, c2 in enumerate(candidates):        
            if i == j:
                result[i].append(None)
                continue
            try:
                ce = mmap[c1][c2]
            except:
                ce = generate_counter_example(c1,c2)
            result[i].append(ce)
    return result


# the strategy to pick the most dividing example should be implemented here
def pick_example(row, candidates, match_dict, examples_matrix, truth):
    # pick the most dividing example i.e.  the one which has a div_list that is
    # nearest to candidates/2
    scores = {}
    cand_cnt = len(candidates)
    total_scores = 0
    for idx, candidate in enumerate(candidates):
        scores[candidate] = cand_cnt - idx
        total_scores += scores[candidate]
    logging.debug("scores map: " + str(scores))
    max_score = -1
    picked_example = ""
    picked_div_list = []
    for e, acc_bucket in match_dict.items():
        rej_bucket = diff(candidates, acc_bucket)
        acc_scores = ([scores[x] for x in acc_bucket]) 
        rej_scores = ([scores[x] for x in rej_bucket]) 
        #pr1 = (max([x/sum(acc_scores) for x in acc_scores])) if len(acc_bucket) > 0 else -100
        pr1 = acc_scores[0]  if len(acc_bucket) > 0 else -1000
        #pr2 = (max([x/sum(rej_scores) for x in rej_scores])) if len(rej_bucket) > 0 else -100
        pr2 = rej_scores[0] if len(rej_bucket) > 0 else -1000
        current_score = (pr1 + pr2) / 2
        logging.debug("for example "+ e + " we have the following score expectation: "+ str(current_score))
        if  current_score >= max_score:
            picked_example = e
            picked_div_list = acc_bucket
            max_score = current_score
    return picked_example, picked_div_list


# interact with the user (once) and return a shorter list of candidates
def prune_candidates(row, interaction,  truth, candidates):
    result = []
    examples_matrix = load_example_matrix(candidates)
    logging.debug("example matrix: \n" + str(np.matrix(examples_matrix)))
    # identify accepting buckets for each example
    match_dict = identify_devisions(row, candidates, examples_matrix)
    logging.debug("match dictionary: \n" + str(np.matrix(match_dict)))
    # pick the best example
    picked_example, picked_div_list = pick_example(row, candidates, match_dict, examples_matrix, truth)
    # check which bucket to select
    is_positive = util.is_match_rfixer(row, truth, picked_example)
    logging.debug("is the picked example " + picked_example + " matched by ground truth " + truth + "  " + str(is_positive))
    # prune candidates
    if is_positive:
        result = util.comm_ordered_list(candidates, picked_div_list)
    else:
        result = util.diff_ordered_list(candidates, picked_div_list)
    logging.debug("pruned candidates: " + str(result))
    return result


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
    for key, value in res.items():
        tr_cnt = len([x for x in value if x is True])
        print("success rate with " + key + " examples: " + str((tr_cnt) / len(value) * 100) + "%")
    print("# total running time: %s seconds" % (time.time() - start_time))

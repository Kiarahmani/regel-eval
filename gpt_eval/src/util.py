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

# set up config file
dirname = os.path.dirname(__file__)
conf_file = os.path.join(dirname, './config.yaml')
conf = yaml.safe_load(open(conf_file))

match_file = os.path.join("cache", conf['cache']['match_file'])
example_file = os.path.join("cache", conf['cache']['example_file'])


# initialize logger
logging.basicConfig(format='[%(levelname)s] [%(asctime)s] [%(funcName)-10s%(lineno)s]  %(message)s', level=logging.ERROR)



# get file encoding type
def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


# run a command with a timeout
def run(cmd, timeout_sec):
    start_time = time.time()
    proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    ret = 1
    logging.debug("running external command: "+ cmd)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        ret = proc.returncode
        #logging.debug(stdout.decode('Latin-1'))
    except Exception as e:
        raise e
    finally:
        timer.cancel()
    if (ret != 0):
        logging.debug("program returned non 0: ")
        #logging.debug(stdout.decode('Latin-1'))
        #logging.debug(stderr.decode('Latin-1'))
    return ret, stdout.decode('Latin-1')


# returns items from first list that are not in the second list
# preserves the order of the first list
def diff_ordered_list(first, second):
        second = set(second)
        return [item for item in first if item not in second]


# returns items from first list that are in the second list
# preserves the order of the first list
def comm_ordered_list(first, second):
        second = set(second)
        return [item for item in first if item in second]


# removes duplicate items from a list while preserving the order
def remove_dupl_ordered_list(seq):
    return pd.unique(seq)
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



# given a list of regexes returns a sublist of semantically distinct regexes
def remove_dupl_regex_semantic(regs):
    result = []
    regs = list(set(regs))
    for reg in regs:
        add = True
        for res in result:
            if compare(reg,res):
                add = False
                break
        if add:
            result.append(reg)
    return result







# Given the correct regex, attempt repair of a faulty regex within specified
# time out.  Returns the numnber of required machine-generated examples
def attempt_repair(regex, truth):
    cmd = conf['command']['repair'] + ' --r1 \'"' + regex + '"\' --r2 \'"' + truth + '"\''
    ret, rf_out = run (cmd, conf['command']['timeout'])
    #print (rf_out)
    r_init_pos = r'That that should match the strings:[\s\r\n]+\?\s\(.+:.+\)\s+(.*)'
    r_init_neg = r'And reject the strings:[\s\r\n]+\?\s\(.+:.+\)\s+(.*)'
    r_neg = r'add negative:\s+(.){1,2}[$\r]'
    r_pos = r'add positive:\s+(.){1,2}[$\r]'
    r_solution = r'Finds the following solutions \(and the corresponding fitness\):\s*\d\s*(.*)\s*All done'
    s_solution = re.search(r_solution,rf_out)
    s_init_pos = re.search(r_init_pos,rf_out)
    s_init_neg = re.search(r_init_neg,rf_out)
    final_neg = re.findall(r_neg, rf_out)
    final_pos = re.findall(r_pos, rf_out)
    final_solution = None
    if s_solution != None:
        final_solution = s_solution.group(1)
    if s_init_pos != None:
        final_pos.append(s_init_pos.group(1))
    if s_init_neg != None:
        final_neg.append(s_init_neg.group(1))
    if (ret == 0):
        return final_solution, final_pos, final_neg
    else:
        return None, None, None
    




# compare two regex: return true if semantically equivalent
def compare_regex_semantic(r1, r2):
    if r1 == None or r2 == None:
        return False
    #logging.debug("comparing " + r1 + " with " + r2)
    #print (r1 + "==" + r2)
    if (r1 == r2):
        return True
    try:
        cmdc = conf['command']['compare']
        args = ' --r1 \'"' + r1 + '"\' --r2 \'"' + r2 + '"\''
        ret, rfout = run (cmdc + args, 10)
        if ret == 0:
            res = ('equivalent' in rfout)
            logging.debug(res)
            return res
        else:
            return False
    except Exception as inst:
            raise inst


def is_match_python(regex, example):
    try:
        res = len(re.findall(regex,example)) > 0
    except Exception as ex:
        print(ex)
        return False
    return res






def is_match_rfixer_help(regex, example):
    try:
        cmdc = conf['command']['match']
        args = ' --r \'"' + regex + '"\' --e \'"' + example + '"\''
        ret, rfout = run(cmdc + args, 10)
        if ret == 0:
            return ('true' in rfout and 'accepts' in rfout)
        else:
            return False
    except Exception as inst:
            return False



def is_match_rfixer(regex, example):
    if conf['cache']['should_use']:
        # first try to look up the match from the cache file
        with open(match_file, 'r', encoding="utf-16") as file:
            old_map = {}
            for line in file:
                if line:
                    old_map = json.loads(line)
                    break
            if regex in old_map.keys():
                entry = old_map[regex]
                if example in entry[0]:
                    return True
                elif example in entry[1]:
                    return False
                # regex key exists but example is neither in accepted or rejected lists
                m = is_match_rfixer_help(regex, example)
                if m:
                    old_map[regex][0].append(example)
                else:
                    old_map[regex][1].append(example)
            else:
                # regex does not exists: initialize
                m = is_match_rfixer_help(regex, example)
                lt = [example] if m else []
                lf = [example] if not m else []
                old_map[regex] = [lt,lf]
        with open(match_file, 'w', encoding="utf-16") as file:
            file.write(json.dumps(old_map))
        return m

   





def generate_counter_example_rfixer(r1, r2):
    if (r1 == r2):
        logging.debug(r1 + " and " + r2 + " are syntactically equivalent and no counter example exists")
        return None
    try:
        cmdc = conf['command']['compare']
        args = ' --r1 \'"' + r1 + '"\' --r2 \'"' + r2 + '"\''
        ret, rfout = run (cmdc + args, 10)
        #print (rfout)        
        if ret == 0:
            reg = r'A string accepted by r1 and rejected by r2: (.*)'
            if not 'A string accepted' in rfout:
                logging.debug("RFixer returned without an example. no counter example was generated")
                return None
            ex = re.findall(reg,rfout)[0]#[:-1]
            #print ("ex: " + ex)   
            if 'null' in ex or 'A string accepted' in ex:
                #print ("RFixer returned without an example. no counter example was generated")
                #print (rfout)
                return None
            return  ex
        else:
            print("RFixer call returned error code. no counter-example was generated")
            print (rfout)
            return None
    except Exception as inst:
            print("an exception occured when calling RFixer. no counter example was generated")
            raise inst







# generate counter example that is accepted by r1 and is rejected by r2: return
# None if no such example exists
def generate_counter_example(r1, r2):
    if conf['cache']['should_use']:
        # first try to look up the example from the cache file
        with open(example_file, 'r', encoding="utf-16") as file:
            old_map = {}
            for line in file:
                if line:
                    old_map = json.loads(line)
                    break
            if r1 in old_map.keys():
                if r2 in old_map[r1].keys():
                    return old_map[r1][r2]
                else: 
                    ce = generate_counter_example_rfixer(r1, r2)
                    if ce == None:
                        ce = "None"
                    old_map[r1][r2] = ce
            else:
                old_map[r1] = {}
                ce = generate_counter_example_rfixer(r1, r2)
                if ce == None:
                    ce = "None"
                old_map[r1][r2] = ce
        with open(example_file, 'w', encoding="utf-16") as file:
            file.write(json.dumps(old_map))


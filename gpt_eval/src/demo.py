import util
import os
import yaml
import re
import logging

# set up config file
dirname = os.path.dirname(__file__)
conf_file = os.path.join(dirname, './config.yaml')
conf = yaml.safe_load(open(conf_file))


# F1
# takes a regex and two lists of positive and negative examples and returns 
# a regex which is consistent with all examples
def repair_regex(regex, positives, negatives):
    with open('temp.rfixer', 'w', encoding='utf-8') as f:
        f.write(str(regex) + "\n")
        f.write('+++\n')
        for p in positives:
            f.write(p + "\n")
        f.write('---\n')
        for n in negatives:
            f.write(n + "\n")
    ret, rf_out = util.run(conf['command']['fix'] + " ./temp.rfixer ", conf['command']['timeout'])
    r = r'Finds the following solutions \(and the corresponding fitness\):\s*\d\s*(.*)\s*All done'
    result = re.search(r,rf_out)
    if (result != None):
        return result.group(1)
    else:
        return None
   

# F2
# takes a list of regexes and returns a string (i.e. example) to be prompted to the users
def generate_test_case(candidates):
    result = ""
    scores = {}
    example_dict = generate_examples_dict(candidates)
    for idx, candidate in enumerate(candidates):
        scores[candidate] = len(candidates) - idx
    for e, acc_bucket in example_dict.items():
        rej_bucket = util.diff_ordered_list(candidates, acc_bucket)
        pr1 = scores[acc_bucket[0]] if len(acc_bucket) > 0 else - 1000
        pr2 = scores[rej_bucket[0]] if len(rej_bucket) > 0 else - 1000
        max_score = -1 
        if  pr1 + pr2 >= max_score:
            result = e
            max_score = pr1 + pr2
    return result
   



# takes a list of regexes and returns a list of potential test cases (i.e. example) to be prompted to the users
def generate_potential_test_case(candidates):
    result = []
    example_dict = generate_examples_dict(candidates)
    for e, acc_bucket in example_dict.items():
        if e != None and e!= 'None':
            result.append(e)
    return result


# F3
# takes a list of candidate regexes, a test string and a boolean that shows if the test is matched by 
# the ground truth or not. Returns a smaller list of candidates which are consistent with the provided test case
def filter_regex_list(candidates, test_case, is_matched):
    res = []
    for candidate in candidates:
        if util.is_match_rfixer_help (candidate, test_case) == is_matched:
            res.append(candidate)
    return res



# given a list of candidates, generates a map from examples to the list of candidates which accept them
def generate_examples_dict(candidates):
    result = {}
    logging.debug("initial empty dict: " + str(result))
    for i, c1 in enumerate(candidates):
        for j, c2 in enumerate(candidates):
            if i == j:
                continue
            ce = util.generate_counter_example(c1,c2)
            logging.debug("generating a counter-example for: " + c1 + " and " + c2)
            logging.debug("counter example: " + str(ce))
            if ce != None:
                if ce in result.keys():
                    result[ce].append(c1)
                else:
                    result[ce] = [c1]
            #logging.debug("current example dict: " + str(result))
    return result
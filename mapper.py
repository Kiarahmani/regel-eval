import os
import re
import pandas as pd

def read_file(filename):
  with open(filename, 'r') as f:
      s = f.read()
  marker1 = "// natural language"
  marker2 = "// examples"
  marker3 = "// gt"
  assert s.find(marker1) != -1, "Marker {0} not found in file".format(marker1)
  index1 = s.find(marker1) + len(marker1)
  index1end = s.find(marker2)
  if index1end == -1:
      marker2 = "// example"
      index1end = s.find(marker2)
  assert index1end != -1, "Marker {0} not found in file".format(marker2)
  index2 = index1end + len(marker2)
  index2end = s.find(marker3)
  assert index2end != -1, "Marker {0} not found in file".format(marker3)
  index3 = index2end + len(marker3)
  nl = s[index1:index1end].strip()
  ex = s[index2:index2end].strip()
  marker4 = "// visual"
  index4 = s.find( marker4 )
  gt = s[index3:].strip() if index4 == -1 else s[index3:index4].strip()
  parsed_gt = parse(gt)
  return (nl, ex, parsed_gt)

ops = ["concat","contain","or","and","startwith", "endwith", "star", "optional", "notcc", "not", "repeatatleast", "repeatrange", "repeat"]
term = {"<num>": "0-9", "<num1-9>": "1-9", "<let>": "a-zA-Z", "<cap>": "A-Z", "<low>": "a-z", "<vow>": "AEIOUaeiou", 
        "<hex>": "0-9A-Fa-f", "<alphanum>": "0-9A-Za-z", "<spec>": "-!@#$%^&*()_.", "<any>": ".", "</>": "/",
        "<\\>": '"\\"', "<|>": "\\|", "<&>": "\\&", "<(>": "\\(", "<)>": "\\)", "<?>": "\\?",
        "<.>": "\\.", "<,>": ",", "< >": " ", "<->": "\\-", "<_>": "_",
        "<+>": "\\+", "<*>": "\\*", "<m2>": "#", "<=>": "=", "<^>": "\\^",
        "<;>": ";", "<:>": ":", "<%>": "%", "<m1>": "@", "<<>": "<", "<>>": ">",
        "<m0>": "!", "<m3>": "\\$", "<~>": "~", "<{>": "{", "<}>": "}", "<@>": "@", "<!>": "!", "<#>": "#" }

def to_string_terminal(regex):
    assert isinstance(regex, str), "Expected regex {0} to be a string".format(regex)
    ret_str = term.get(regex)
    if ret_str:
        return "[{0}]".format(ret_str)
    # regex can be a number
    regex_new = regex[1:-1] if regex.startswith("<") and regex.endswith(">") else regex
    if regex_new.isnumeric() or regex_new.isalpha():
        return regex_new
    else:
        assert False, "Unknown constant {0}".format(regex)

def to_string( regex ):
    if not isinstance(regex, list):
        return to_string_terminal(regex)
    func = regex[0]
    if func == "concat":
        s1 = to_string(regex[1])
        s2 = to_string(regex[2])
        return s1 + s2
    if func == "contain":
        s1 = to_string(regex[1])
        return ".*" + s1 + ".*"
    if func == "or":
        s1 = to_string(regex[1])
        s2 = to_string(regex[2])
        return "(" + s1 + ")|(" + s2 + ")"
    if func == "startwith":
        s1 = to_string(regex[1])
        #return "^" + s1
        return s1 + ".*"
    if func == "endwith":
        s1 = to_string(regex[1])
        return ".*" + s1
    if func == "star":
        s1 = to_string(regex[1])
        return "(" + s1 + ")*"
    if func == "optional":
        s1 = to_string(regex[1])
        return "(" + s1 + ")?"
    if func == "not":
        s1 = to_string(regex[1])
        return "!({0})".format(s1)
    if func == "repeatatleast":
        s1 = to_string(regex[1])
        s2 = to_string(regex[2])
        return "({1}){{{0},}}".format(s2, s1)
    if func == "repeat":
        s1 = to_string(regex[1])
        s2 = to_string(regex[2])
        return "({1}){{{0}}}".format(s2, s1)
    if func == "repeatrange":
        s1 = to_string(regex[1])
        s2 = to_string(regex[2])
        s3 = to_string(regex[3])
        return "({1}){{{0},{2}}}".format(s2, s1, s3)
    # not, notcc, and
    if func in ["not", "notcc", "and"]:
        assert False, "Cant handle {0}".format(func)
    else:
        assert False, "Unexpected function {0}".format(func)

def parse(gt, stack=[]):
    gt = gt.strip()
    if gt == "":
        assert len(stack) == 1, "Stack should have 1 element: {0}".format(stack)
        # print("Parsed as {0}".format(stack[0]))
        return stack.pop()
    for i in ops:
        if gt.startswith(i):
            stack.append(i)
            return parse(gt[len(i):], stack)
    m = re.match(r'^\d+', gt)
    if m:
        index_end = m.span()[1]
        stack.append(gt[:index_end])
        return parse(gt[index_end:], stack)
    if gt.startswith("<"):
        index_end = gt.find(">")
        assert index_end != -1, "< is not followed by > in {0}".format(gt)
        stack.append(gt[:index_end+1])
        return parse(gt[index_end+1:], stack)
    if gt.startswith("("):
        stack.append("(")
        return parse(gt[1:], stack)
    if gt.startswith(")"):
        new_term = []
        a = stack.pop()
        while a != "(":
            new_term.append(a)
            a = stack.pop()
        a = stack.pop()
        new_term.append(a)
        new_term.reverse()
        stack.append(new_term)  # FIX HERE
        return parse(gt[1:], stack)
    if gt.startswith(","):
        return parse(gt[1:], stack)
    assert False, "Found unexpected character {0}".format(gt)

def process_examples(ex):
    examples = ex.split("\n")
    examples = [e.strip() for e in examples if e.strip()]
    pos_examples = [e[:-2].strip() for e in examples if e.strip().endswith("+")]
    neg_examples = [e[:-2].strip() for e in examples if e.endswith("-")]
    return pos_examples, neg_examples

if __name__ == "__main__":
    dir_paths = ["../../../../exp/so/benchmark/", "../../../../exp/deepregex/benchmark/"]
    for dir_path in dir_paths:
      files = os.listdir( dir_path )
      dflist = []
      for f in files:
        # print("Processing File {0}".format(f))
        file_path = dir_path + f
        if os.path.isdir(file_path) or f.startswith(".") or f.endswith("~"):
            continue
        nl, ex, gt = read_file(file_path)
        #print("File {0} done".format(f))
        try:
            regex_str = to_string(gt)
            pos_ex, neg_ex = process_examples(ex)
            dflist.append( [nl, regex_str, " ".join(pos_ex), " ".join(neg_ex)] )
        except AssertionError as e:
            print("Ignoring benchmark {0} due to {1}".format(f, e))
      index1 = dir_path.find("exp")
      index2 = dir_path.find("benchmark")
      s = ["../../../../exp/so/benchmark/", "../../../../exp/deepregex/benchmark/"]
      df = pd.DataFrame(dflist)
      df.to_csv(dir_path[index1+4:index2-1] + ".csv", index=False)


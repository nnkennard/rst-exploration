import json
import os
import re
import subprocess
import uuid


class Node(object):
  def __init__(self, label=None, nucleus=None, satellite=None, text=None):
    self.nucleus = nucleus
    self.satellite = satellite
    self.text = text
    self.label = label

def tokenize(program):
  return re.findall('[\(\)]|[A-Za-z\:\-]+|[0-9]+', program)

def read_from_tokens(tokens):
  if len(tokens) == 0:
    raise SyntaxError('unexpected EOF')
  token = tokens.pop(0)
  if token == '(':
    L = []
    while tokens[0] != ')':
      L.append(read_from_tokens(tokens))
    tokens.pop(0) # pop off ')'
    return L
  elif token == ')':
    raise SyntaxError('unexpected )')
  else:
    return atom(token)

def atom(token):
    "Numbers become numbers; every other token is a symbol."
    try: return int(token)
    except ValueError:
      return token

def parse(program):
  return read_from_tokens(tokenize(program))
  
def build_tree(tree_list, texts):
  if len(tree_list) == 2:
    label, node = tree_list
    if label == "text":
      return Node(text=texts[node-1])
    else:
      return Node(nucleus=build_tree(node, texts), label=label)
  else:
    assert len(tree_list) == 3
    label, nucleus, satellite = tree_list
    return Node(nucleus=build_tree(nucleus, texts),
        satellite=build_tree(satellite, texts), label=label)
  return Node

def rst_parse(texts):
  identifier = str(uuid.uuid4())

  filenames = []
  for i, text in enumerate(texts):
    tmp_filename = f'tmp/{identifier}_{i}.txt'
    filenames.append(tmp_filename)
    with open(tmp_filename, 'w') as f:
      f.write(text)

  proc = subprocess.check_output([
      "rst_parse", "-g", "segmentation_model.C1.0", "-p",
      "rst_parsing_model.C1.0"] + filenames)

  for filename in filenames:
    subprocess.run(["rm", filename])

  trees = []
  for pre_res in proc.decode('utf-8').strip().split("\n"):
    result = json.loads(pre_res)
    texts = result['edu_tokens']
    trees.append(build_tree(parse(result['scored_rst_trees'][0]['tree']),
      texts))
    
  return trees

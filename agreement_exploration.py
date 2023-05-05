import collections
import glob
from lisp_lib import parse

PATH_PREFIX = "rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/"

def list_jaccard(l1, l2):
  if not set(l1).union(l2):
    return 0.0
  return len(set(l1).intersection(l2))/len(set(l1).union(l2))

def get_edu_list(parse):
    edus = [None]

    def inorder_walk(tree):
        current_edu = []
        for i, x in enumerate(tree):
            if type(x) in [int, float, str]:
                if tree[0] == "text" and i > 0:
                    str_x = str(x)
                    p = str_x.replace("_!", "")
                    if str_x.startswith("_!"):
                        current_edu.append(p)
                    else:
                        current_edu.append(p)
                    if str_x.endswith("_!"):
                        edus.append(current_edu)
                        current_edu = []
            else:
                inorder_walk(x)

    inorder_walk(parse)
    return edus

def get_span_list(parse):
    spans = collections.defaultdict(list)

    def inorder_walk(tree):
        for i, x in enumerate(tree):
          if type(x) == list:
            if x[0] == 'span':
              diff = x[2] - x[1]
              spans[diff].append(tuple(x[1:]))
              #spans.append(tuple(x[1:]))
            inorder_walk(x)


    inorder_walk(parse)
    return spans



def compare(a,b):
  tree_a = parse(a)
  tree_b = parse(b)
  edu_a =[" ".join(x) for x in  get_edu_list(tree_a)[1:]]
  edu_b =[" ".join(x) for x in  get_edu_list(tree_b)[1:]]
  a_spans = get_span_list(tree_a)
  b_spans = get_span_list(tree_b)

  for index in sorted(set(a_spans.keys()).union(b_spans.keys())):
    print(index, list_jaccard(a_spans[index], b_spans[index]))


  #print(get_span_list(tree_a))

  #print(len(set(edu_a).intersection(edu_b))/len(set(edu_a).union(edu_b)))
  #print(len(set(edu_a).union(edu_b)) - len(set(edu_a).intersection(edu_b)),
  #len(edu_a), len(edu_b))



def main():

  training_files = [l.split("/")[-1] for l in
  glob.glob(f'{PATH_PREFIX}TRAINING/*.out')]
  double_files = [l.split("/")[-1] for l in
  glob.glob(f'{PATH_PREFIX}/*.out'.replace('main', 'double'))]


  match = 0
  mismatch = 0
  train_doubles = list(sorted(set(training_files).intersection(double_files)))
  for x in train_doubles:
    with open(f'{PATH_PREFIX}/TRAINING//{x}', 'r') as f:
      j = f.read()
      with open(f'{PATH_PREFIX}/TRAINING/{x}.dis', 'r') as g:
        a = g.read()
      with open(f'{PATH_PREFIX}/{x}.dis'.replace('main', 'double'), 'r') as g:
        b = g.read()

    compare(a,b)

  print(match, mismatch)

if __name__ == "__main__":
  main()


import glob

import lisp_lib

TRAINING_PATH = "rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/*.dis"

def pretty_print(tree, indent=0):
  for x in tree:
    if type(x) in [str, int, float]:
      print(" "*indent + str(x), end="")
      if type(x) == str and tree[0] == 'text':
        if x.startswith("_!"):
          print("B")
        else:
          print("I")
      else:
        print()
    else:
      pretty_print(x, indent+1)
    print()

def main():

  for filename in glob.glob(TRAINING_PATH):
    with open(filename, 'r') as f:
      text = f.read()
      parsed = lisp_lib.parse(text)
      pretty_print(parsed)
      print()



if __name__ == "__main__":
  main()


import glob

import lisp_lib

TRAINING_PATH = "rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/*.dis"

def pretty_print(tree, indent=0):
  for i, x in enumerate(tree):
    if type(x) in [str, int, float]:
      #print(" "*indent + str(x), end="")
      if type(x) == str and tree[0] == 'text' and i > 0:
        if x.startswith("_!"):
          print(f"{x}\tB")
        else:
          print(f"{x}\tI")
      else:
        print()
    else:
      pretty_print(x, indent+1)
    print()


def print_edus(tree, indent=0):
  for i, x in enumerate(tree):
    if type(x) in [str, int, float]:
      #print(" "*indent + str(x), end="")
      if type(x) == str and tree[0] == 'text' and i > 0:
        p = x.replace("_!", "")
        if x.startswith("_!"):
          print(f"\n{p} ", end='')
        else:
          print(f"{p} ", end='')
    else:
      print_edus(x, indent+1)
    #print("~")


def main():

  for filename in glob.glob(TRAINING_PATH):
    with open(filename, 'r') as f:
      text = f.read()
      parsed = lisp_lib.parse(text)
      print_edus(parsed)
      print()



if __name__ == "__main__":
  main()


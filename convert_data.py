import glob

import lisp_lib

TRAINING_PATH = "rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/*.dis"


def pretty_print(tree, indent=0):
    for i, x in enumerate(tree):
        if type(x) in [str, int, float]:
            # print(" "*indent + str(x), end="")
            if type(x) == str and tree[0] == "text" and i > 0:
                if x.startswith("_!"):
                    print(f"{x}\tB")
                else:
                    print(f"{x}\tI")
            else:
                print()
        else:
            pretty_print(x, indent + 1)
        print()


def print_edus(tree, indent=0):
    for i, x in enumerate(tree):
        if type(x) in [str, int, float]:
            # print(" "*indent + str(x), end="")
            if type(x) == str and tree[0] == "text" and i > 0:
                p = x.replace("_!", "")
                if x.startswith("_!"):
                    print(f"\n{p} ", end="")
                else:
                    print(f"{p} ", end="")
        else:
            print_edus(x, indent + 1)
        # print("~")


def convert_to_segmentation_format(parsed):
    pass


def get_edu_list(parse):
    edus = [None]

    def inorder_walk(tree):
        current_edu = []
        for i, x in enumerate(tree):
            if type(x) in [int, float]:
                continue
            elif type(x) == str:
                if tree[0] == "text" and i > 0:
                    p = x.replace("_!", "")
                    if x.startswith("_!"):
                        current_edu.append(p)
                    else:
                        current_edu.append(p)
                    if x.endswith("_!"):
                        edus.append(current_edu)
                        current_edu = []
            else:
                inorder_walk(x)

    inorder_walk(parse)

    for i, y in enumerate(edus):
        print(i, y)


class GoldRSTParse(object):
    def __init__(self, parse_text):
        parse = lisp_lib.parse(parse_text)
        edu_list = get_edu_list(parse)
        pass


def main():
    for filename in glob.glob(TRAINING_PATH):
        with open(filename, "r") as f:
            text = f.read()
            _ = GoldRSTParse(text)
            # parsed = lisp_lib.parse(text)
            # convert_to_segmentation_format(parsed)
            # print_edus(parsed)
            # print()


if __name__ == "__main__":
    main()

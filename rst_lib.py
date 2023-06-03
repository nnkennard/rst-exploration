import json
import lisp_lib
import collections
import glob

# ========== Utils and constants ==================================================

AnnotationPair = collections.namedtuple(
    "AnnotationPair", "identifier input_text main_annotation double_annotation"
)


TRAIN, TEST, DOUBLE, SINGLE = "train test double single".split()

DATA_PATH = "./rst_discourse_treebank/"


# ========== RST-DT file helpers ==================================================
# These are functions that deal with the particulars of RST-DT's format

def bracket_conversion(parse_text):
    if "TT_ERR" in parse_text:
        return None
    in_text = False
    new_text = ""
    for i, c in enumerate(parse_text):
        if in_text:
            new_text += c.replace("(", "-LRB-").replace(")", "-RRB-")
            if c == "_" and parse_text[i + 1] == "!":
                in_text = False
        else:
            if c == "_" and parse_text[i + 1] == "!":
                in_text = True
            new_text += c

    return new_text


def read_tag(parse):
    if parse[0] == "span":
        return {"span": tuple(int(x) for x in parse[1:])}
    elif parse[0] == "text":
        assert parse[1].startswith("_!") and parse[-1].endswith("_!")
        tokens = [str(i) for i in parse[1:]]
        tokens[0] = tokens[0][2:]
        tokens[-1] = tokens[-1][:-2]
        return {"text": " ".join(tokens), "tokens": tokens}
    elif parse[0] == "leaf":
        return {"leaf": int(parse[1])}
    else:
        assert parse[0] == "rel2par"
        return {parse[0]: parse[1]}
    

def is_tag(parse):
    return all(type(x) in [str, float, int] for x in parse)


def get_tags(parse):
    tags = {}
    for maybe_tag in parse:
        if is_tag(maybe_tag):
            tags.update(read_tag(maybe_tag))
    return tags

# ========== RST-DT directory helpers ==================================================
# These are functions that deal with the particulars of RST-DT's directory structure


def get_file_pair(path, input_file):
    with open(f"{path}{input_file}", "r") as f:
        input_text = f.read()
    with open(f"{path}{input_file}.dis", "r") as f:
        output_text = f.read()
    return input_text, output_text

def build_file_map(data_path=DATA_PATH):
    main_path = f"{data_path}/data/RSTtrees-WSJ-main-1.0/"
    double_path = f"{data_path}/data/RSTtrees-WSJ-double-1.0/"
    paths = {
        TRAIN: f"{main_path}TRAINING/",
        TEST: f"{main_path}TEST/",
        DOUBLE: double_path,
    }
    temp_file_map = collections.defaultdict(list)
    for subset, path in paths.items():
        for filename in glob.glob(f"{path}*.out") + glob.glob(f"{path}file?"):
            identifier = filename.split("/")[-1].split(".")[0]
            temp_file_map[subset].append(identifier)

    files = collections.defaultdict(lambda: collections.defaultdict(list))
    for subset in [TRAIN, TEST]:
        for filename in temp_file_map[subset]:
            if filename in temp_file_map[DOUBLE]:
                files[subset][DOUBLE].append(filename)
            else:
                files[subset][SINGLE].append(filename)

    return paths, files

def build_annotation_pair(files, paths, identifier):
    if identifier.startswith("file"):
        input_file = identifier
    else:
        input_file = f"{identifier}.out"
    main_in, main_out = get_file_pair(paths[TRAIN], input_file)
    double_in, double_out = get_file_pair(paths[DOUBLE], input_file)
    assert main_in == double_in
    if None in [main_out, double_out]:
        return
    return AnnotationPair(identifier, main_in, Parse(main_out), Parse(double_out))


# =========== Objects for parse representation =============================

class Subtree(object):
    def __init__(self, parse):
        self.tags = get_tags(parse[1:])
        self.node_type = parse[0][0]
        children = [subtree for subtree in parse if not is_tag(subtree)]

        # Build a right-branching binary tree
        if children:
            self.is_leaf = False
            self.left_child = Subtree(children[0])
            if len(children) == 2:
                self.right_child = Subtree(children[1])
            else:
                assert len(children) > 2
                self.right_child = Subtree(["Nucleus"] + children[1:])
            self.direction = f"{self.left_child.node_type}{self.right_child.node_type}"
            if self.direction == "NS":
                self.relation = self.right_child.tags["rel2par"]
            else:
                self.relation = self.left_child.tags["rel2par"]

        else:
            self.is_leaf = True
            assert "text" in self.tags


class Parse(object):
    def __init__(self, parse_text):
        self.complete = False
        maybe_converted = bracket_conversion(parse_text)
        self.is_valid = maybe_converted is not None
        if self.is_valid:
            parse = lisp_lib.parse(maybe_converted)
            self.tree = Subtree(parse)
            self._assign_span_indices()
            self.edus = self._read_in_order()
            self.complete = True

    def _read_in_order(self):
        edus = []

        def inorder_helper(tree):
            if "tokens" in tree.tags:
                edus.append(tree.tags["tokens"])
            else:
                inorder_helper(tree.left_child)
                inorder_helper(tree.right_child)

        inorder_helper(self.tree)
        return [None] + edus

    def _assign_span_indices(self):
        token_index = [0]  # This is a terrible solution

        def assign_span_helper(subtree):
            if subtree.is_leaf:
                subtree.start_token = token_index[0]
                subtree.end_token = token_index[0] + len(subtree.tags["tokens"])
                token_index[0] += len(subtree.tags["tokens"])
            else:
                assign_span_helper(subtree.left_child)
                subtree.start_token = subtree.left_child.start_token
                assign_span_helper(subtree.right_child)
                subtree.end_token = subtree.right_child.end_token

        assign_span_helper(self.tree)

    def get_span_map(self, edus=None):
        span_map = {}

        def get_span_helper(subtree):
            if not subtree.is_leaf:
                span_map[(subtree.start_token, subtree.end_token)] = (subtree.direction, subtree.relation)
                get_span_helper(subtree.left_child)
                get_span_helper(subtree.right_child)
            else:
                span_map[(subtree.start_token, subtree.end_token)] = (None, None)

        get_span_helper(self.tree)
        if edus is not None:
            print(sorted(set(edus) - set(span_map.keys())))
        return span_map

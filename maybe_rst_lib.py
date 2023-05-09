import collections
import json
import lisp_lib
import glob


def is_tag(parse):
    return all(type(x) in [str, float, int] for x in parse)


def read_tag(parse):
    if parse[0] == "span":
        return {"span": tuple(int(x) for x in parse[1:])}
    elif parse[0] == "text":
        return {"text": parse[1:]}
    elif parse[0] == "leaf":
        return {"leaf": int(parse[1])}
    else:
        assert parse[0] == "rel2par"
        return {parse[0]: parse[1]}


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


def get_span_map(tree):
    span_map = {}

    def span_walk(tree):
        if tree is None or tree.is_leaf:
            return
        if len(tree.satellites) > 1:
            # Multiple satellites in:
            #   TRAINING/wsj_1322 in (129, 144)
            #   TRAINING/wsj_0681 in (110, 115)
            #   TRAINING/wsj_1377 in (116, 157)
            #   TRAINING/wsj_1170 in (4, 18)
            #   TRAINING/wsj_1192 in (73, 75)
            #   TRAINING/wsj_1138 in (32, 37)
            #   TRAINING/wsj_1355 in (1, 5)
            #   TRAINING/wsj_1362 in (12, 14)
            #   TRAINING/wsj_1318 in (15 25
            return
        elif len(tree.nuclei) == 1:
            (nucleus,) = tree.nuclei
            (satellite,) = tree.satellites
            assert nucleus.tags["rel2par"] == "span"
            span_map[(tree.first_edu, tree.last_edu)] = satellite.tags["rel2par"]
        else:
            assert not tree.satellites
            rel2pars = set()
            spans = []
            for c in tree.nuclei:
                rel2pars.add(c.tags["rel2par"])
                spans.append(c.tags["span_leaf"])
            rel2pars -= set(
                ["span"]
            )  # TEST/1189.out.dis has a multinuclear rel where one rel2par is span
            assert len(rel2pars) == 1
            (rel2par,) = sorted(rel2pars)
            span_map[(tree.first_edu, tree.last_edu)] = rel2par

        for s in tree.nuclei + tree.satellites:
            span_walk(s)

    span_walk(tree)
    return span_map


class Parse(object):
    def __init__(self, parse_text):
        temp = bracket_conversion(parse_text)
        if temp is None:
            self.is_valid = False
            return
        else:
            self.is_valid = True
        parse = lisp_lib.parse(temp)
        self.tree = Subtree(parse)
        self.edus = get_edu_list(parse)
        self.span_map = get_span_map(self.tree)


class Subtree(object):
    def __init__(self, parse):
        self.tree_text = json.dumps(parse)
        label = parse[0]
        self.tags = {"is_root": label == "Root"}

        children = {"Nucleus": [], "Satellite": []}

        for child in parse[1:]:
            if is_tag(child):
                self.tags.update(read_tag(child))
            else:
                children[child[0]].append(Subtree(child))

        self.nuclei = children["Nucleus"]
        self.satellites = children[
            "Satellite"
        ]  # For some reason one tree has two satellites
        self.tags["span_leaf"] = self.tags.get("span", self.tags.get("leaf"))
        self.is_leaf = "leaf" in self.tags

        edus = set()
        for k in self.nuclei + self.satellites:
            if "span" in k.tags:
                edus.update(k.tags["span"])
            else:
                edus.add(k.tags["leaf"])

        if not self.is_leaf:
            self.first_edu = min(edus)
            self.last_edu = max(edus)


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


TRAIN, TEST, DOUBLE, SINGLE = "train test double single".split()

DATA_PATH = "./rst_discourse_treebank/"


def build_paths(data_path):
    main_path = f"{data_path}/data/RSTtrees-WSJ-main-1.0/"
    double_path = f"{data_path}/data/RSTtrees-WSJ-double-1.0/"
    return {
        TRAIN: f"{main_path}TRAINING/",
        TEST: f"{main_path}TEST/",
        DOUBLE: double_path,
    }


AnnotationPair = collections.namedtuple(
    "AnnotationPair", "input_text main_annotation double_annotation"
)


def build_annotation_pair(input_text, main_annotation_text, double_annotation_text):
    if None in [main_annotation_text, double_annotation_text]:
        return
    return AnnotationPair(
        input_text, Parse(main_annotation_text), Parse(double_annotation_text)
    )


def get_file_pair(path, filename):
    if filename.startswith("file"):
        input_file = filename
    else:
        input_file = f"{filename}.out"

    with open(f"{path}{input_file}", "r") as f:
        input_text = f.read()
    with open(f"{path}{input_file}.dis", "r") as f:
        output_text = bracket_conversion(f.read())
    return input_text, output_text


paths = build_paths(DATA_PATH)

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

annotation_pairs = []

for identifier in files[TRAIN][DOUBLE]:
    main_in, main_out = get_file_pair(f"{paths[TRAIN]}", identifier)
    double_in, double_out = get_file_pair(f"{paths[DOUBLE]}", identifier)
    assert main_in == double_in
    annotation_pairs.append(build_annotation_pair(main_in, main_out, double_out))


def list_jaccard(l1, l2):
    if not (l1 or l2):
        return None
    return len(set(l1).intersection(l2)) / len(set(l1).union(l2))


for x in annotation_pairs:
    if x is None:
        continue
    main_span_map = x.main_annotation.span_map
    double_span_map = x.double_annotation.span_map

span_labels = collections.Counter()
for filename in glob.glob("rst_discourse_treebank/data/RSTtrees-WSJ-*-1.0/*/*.dis"):
    if (
        "1189" in filename
        or "1322" in filename
        or "1318" in filename
        or "1355" in filename
        or "1362" in filename
        or "1138" in filename
        or "1192" in filename
        or "0681" in filename
        or "1377" in filename
        or "1170" in filename
    ):
        continue
    with open(filename, "r") as f:
        parse = Parse(f.read())
        if parse.is_valid:
            for k, v in parse.span_map.items():
                span_labels[v] += 1

import json
import collections
import glob

# ========== Utils and constants ==================================================

TRAIN, TEST, DOUBLE, SINGLE = "train test double single".split()

# ========== Lisp parsing =========================================================


def parse(program):
    "Read a Scheme expression from a string."
    tokenized = program.replace("(", " ( ").replace(")", " ) ").split()
    return read_from_tokens(tokenized)


def read_from_tokens(tokens):
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF")
    token = tokens.pop(0)
    if token == "(":
        L = []
        while tokens[0] != ")":
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ")":
        raise SyntaxError("unexpected )")
    else:
        return token


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


def build_file_map(data_path):
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


# =========== Objects for parse representation =============================

class Span(object):
    """This contains information about the span, except which spans are its children."""
    
    def __init__(self, tags, node_type, subtree, is_leaf):
        self.is_leaf = is_leaf
        self.node_type = node_type
        self.tags = tags
        if not self.is_leaf:
            self.direction = f"{subtree.left_child.span.node_type}{subtree.right_child.span.node_type}"
            if self.direction == "NS":
                self.relation = subtree.right_child.span.tags["rel2par"]
            else:
                self.relation = subtree.left_child.span.tags["rel2par"]


class Subtree(object):
    def __init__(self, parse):
        tags = get_tags(parse[1:])
        node_type = parse[0][0]
        children = [subtree for subtree in parse if not is_tag(subtree)]
        # Build a right-branching binary tree
        if children:
            self.left_child = Subtree(children[0])
            if len(children) == 2:
                self.right_child = Subtree(children[1])
            else:
                assert len(children) > 2
                self.right_child = Subtree(["Nucleus"] + children[1:])

        self.span = Span(tags, node_type, self, is_leaf=len(children) == 0)


class Parse(object):
    def __init__(self, parse_text):
        self.complete = False
        maybe_converted = bracket_conversion(parse_text)
        self.is_valid = maybe_converted is not None
        if self.is_valid:
            parsed = parse(maybe_converted)
            self.tree = Subtree(parsed)
            self._assign_span_indices()
#             self._assign_span_heights()
            self.edus = self._read_in_order()
            self.complete = True

    def _read_in_order(self):
        edus = []

        def inorder_helper(tree):
            if "tokens" in tree.span.tags:
                edus.append(tree.span.tags["tokens"])
            else:
                inorder_helper(tree.left_child)
                inorder_helper(tree.right_child)

        inorder_helper(self.tree)
        return [None] + edus
    
    
#     def _assign_span_heights(self):
        
#         def _assign_height_helper(subtree, parent_depth):
            
#             subtree.span.depth_from_root = parent_depth + 1
#             if subtree.span.is_leaf:
#                 subtree.span.height_from_leaf = 0
#             else:
#                 _assign_height_helper(subtree.left_child, subtree.span.depth_from_root)
#                 _assign_height_helper(subtree.right_child, subtree.span.depth_from_root)
#                 subtree.span.height_from_leaf = max(subtree.left_child.span.height_from_leaf, subtree.right_child.span.height_from_leaf) + 1
        
#         self.tree.span.depth_from_root = 0
#         _assign_height_helper(self.tree, 0)
        
                

    def _assign_span_indices(self):
        token_index = [0]  # This is a terrible solution

        def _assign_indices_helper(subtree):
            if subtree.span.is_leaf:
                subtree.start_token = token_index[0]
                subtree.end_token = token_index[0] + len(subtree.span.tags["tokens"])
                token_index[0] += len(subtree.span.tags["tokens"])
            else:
                _assign_indices_helper(subtree.left_child)
                subtree.start_token = subtree.left_child.start_token
                _assign_indices_helper(subtree.right_child)
                subtree.end_token = subtree.right_child.end_token

        _assign_indices_helper(self.tree)

    def get_span_map(self, edus=None):
        span_map = {}

        def get_span_helper(subtree):
            if not subtree.span.is_leaf:
                span_map[(subtree.start_token, subtree.end_token)] = {
                    "direction": subtree.span.direction,
                    "relation": subtree.span.relation,
#                     "height": subtree.span.height_from_leaf,
#                     "depth": subtree.span.depth_from_root
                }
                get_span_helper(subtree.left_child)
                get_span_helper(subtree.right_child)
            else:
                span_map[(subtree.start_token, subtree.end_token)] = (
                    "Leaf",
                    subtree.span.tags["text"],
                )

        get_span_helper(self.tree)
        
        if edus is not None:
            for additional_span in set(edus) - set(span_map.keys()):
                span_map[additional_span] = ("SegDiff", None)
        
        return span_map

    def render(self, path):
        ete_render(self, path)


class AnnotationPair(object):
    def __init__(self, identifier, input_text, gold_annotation, predicted_annotation, main_is_gold):
        self.identifier = identifier
        self.input_text = input_text
        self.gold_annotation = gold_annotation
        self.predicted_annotation = predicted_annotation
        self.is_valid = (
            self.gold_annotation.is_valid and self.predicted_annotation.is_valid
        )
        if self.is_valid:
            self.final_edus = self._get_final_edus()
            self.gold_span_map = self.gold_annotation.get_span_map(self.final_edus)
            self.predicted_span_map = self.predicted_annotation.get_span_map(self.final_edus)
            self.agreement_scores = self._calculate_agreement_scores()

    def _get_final_edus(self):
        final_edu_ends = sorted(
            set(
                end
                for _, end in set(self.gold_annotation.get_span_map().keys()).union(
                    self.predicted_annotation.get_span_map().keys()
                )
            )
        )
        final_edu_starts = [0] + final_edu_ends[:-1]
        return list(zip(final_edu_starts, final_edu_ends))

    def _calculate_agreement_scores(self):
        f1_scores = {}
        
        set1 = set(self.gold_span_map.keys())
        set2 = set(self.predicted_span_map.keys())
        p = len(set1.intersection(set2)) / len(set1)
        r = len(set1.intersection(set2)) / len(set2)
        f = 2 * p * r / (p + r)
        
        jk = len(set1) + len(set2)
        f1_scores["S"] = f
        
        matched_spans = set1.intersection(set2)
        correct_spans = {
            "N":set(),
            "R":set(),
            "F":set(),
        }
        for span in matched_spans:
            
            rel1 = self.gold_span_map[span]
            rel2 = self.predicted_span_map[span]
            if type(rel1) == tuple or type(rel2) == tuple:
                continue
            
            nuc_match = rel1['direction'] == rel2['direction']
            rel_match = rel1['relation'] == rel2['relation']
            
            if nuc_match and rel_match:
                correct_spans["F"].add(span)
            if nuc_match:
                correct_spans["N"].add(span)
            if rel_match:
                correct_spans["R"].add(span)
           
        for k,v in correct_spans.items():
            f1_scores[k] = 2*len(v)/jk
            
        return f1_scores
                

# =========== Code for handling paired annotations ========================


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
    return [AnnotationPair(
        identifier, main_in, Parse(main_out), Parse(double_out), True
    ),AnnotationPair(
        f'{identifier}_r', main_in, Parse(double_out), Parse(main_out), False
    )
           ]


def get_double_annotated_train_files(data_path, valid_only=False):
    paths, files = build_file_map(data_path)
    pairs = sum([
        build_annotation_pair(files, paths, identifier)
        for identifier in files[TRAIN][DOUBLE]
    ], [])
    if valid_only:
        return [pair for pair in pairs if pair.is_valid]
    else:
        return pairs


# # =========== Wrappers for tree visualization =============================

# from ete3 import NodeStyle, TextFace, Tree, TreeStyle
# import PyQt5

# ts = TreeStyle()
# ts.show_leaf_name = False
# ts.orientation = 1
# ns = NodeStyle()
# ns["size"] = 0


# def get_newick_helper(subtree):
#     if "text" in subtree.tags:
#         return str(subtree.tags["leaf"])
#     else:
#         return f"({get_newick_helper(subtree.left_child)}, {get_newick_helper(subtree.right_child)})"


# def ete_render(annotation, filename):
#     t = Tree(get_newick_helper(annotation.tree) + ";")
#     for x in t.traverse():
#         x.set_style(ns)
#         if x.name:
#             x.add_face(TextFace(f" " + " ".join(annotation.edus[int(x.name)])), 0)
#     t.convert_to_ultrametric()
#     t.render(filename, tree_style=ts, w=400, dpi=150)

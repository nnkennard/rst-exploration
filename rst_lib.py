import json
import collections
import glob

# ========== Scheme parsing =======================================================


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

TRAIN, TEST, DOUBLE, SINGLE = "train test double single".split()

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

# =========== Objects for parse representation =============================

class Subtree(object):
    
    def __init__(self, parse, is_weird_binary=False):
        self.is_weird_binary = is_weird_binary
        self._apply_tags(parse[1:])     
            
        self.nuclearity_contrib = parse[0][0]
        if not self.nuclearity_contrib in "NSR":
            print(parse[0])
            dsds
        
        # Build children
        children = [subtree for subtree in parse if not is_tag(subtree)]
        if children:
            self.is_leaf = False
            self.left_child = Subtree(children[0])
            if len(children) == 2: 
                self.right_child = Subtree(children[1])
            else: # If n-ary, build a right-branching binary tree
                assert len(children) > 2 
                #print(self.left_child.nuclearity_contrib)
                #assert self.left_child.nuclearity_contrib == "N"
                self.right_child = Subtree(["Nucleus", ["rel2par", 'span']] + children[1:], is_weird_binary=True)
                
            # Assign nuclearity
            self.nuclearity = f'{self.left_child.nuclearity_contrib}{self.right_child.nuclearity_contrib}'
            assert self.nuclearity in ['NN', "NS", "SN"]
            if is_weird_binary:
                assert self.nuclearity == "NN"
                
            # Assign relation
            self.relation = self.determine_relation()
        else:
            self.is_leaf = True
            
            
    def _apply_tags(self, parse):
        for maybe_tag in parse:
            if not is_tag(maybe_tag):
                continue
            if maybe_tag[0] == "span":
#                 self.edu_span = tuple(int(x) for x in maybe_tag[1:])
                # Add this on the upward pass instead, so that we can automatically account for the binarization
                self.is_leaf = False
            elif maybe_tag[0] == "leaf":
                leaf_edu = int(maybe_tag[1])
                self.edu_span = (leaf_edu, leaf_edu)
                self.is_leaf = True
            elif maybe_tag[0] == "text":
                assert maybe_tag[1].startswith("_!") and maybe_tag[-1].endswith("_!")
                self.tokens = [str(i).replace("_!", "") for i in maybe_tag[1:]]
            else:
                assert maybe_tag[0] == "rel2par"
                self.rel2par = maybe_tag[1]
                
    def determine_relation(self):
        l = self.left_child.rel2par
        r = self.right_child.rel2par
        assert not l == r == 'span'
        if not l == 'span' and not r == 'span':
            assert l == r
            return l
        elif l == 'span':
            return r
        else:
            return l

class Parse(object):
    def __init__(self, parse_text):
        self.complete = False
        maybe_converted = bracket_conversion(parse_text)
        self.is_valid = maybe_converted is not None
        if self.is_valid:
            parsed = parse(maybe_converted)
            self.tree = Subtree(parsed)
            self._assign_span_indices()
            self.edus = self._read_in_order()
            self.complete = True

    def _read_in_order(self):
        edus = []

        def inorder_helper(tree):
            if tree.is_leaf:
                edus.append(tree.tokens)
            else:
                inorder_helper(tree.left_child)
                inorder_helper(tree.right_child)

        inorder_helper(self.tree)
        return [None] + edus
    
                

    def _assign_span_indices(self):
        token_index = [0]  # This is a terrible solution

        def _assign_indices_helper(subtree):
            print(subtree.is_weird_binary)
            if subtree.is_weird_binary or not subtree.is_leaf:
                _assign_indices_helper(subtree.left_child)
                subtree.start_token = subtree.left_child.start_token
                _assign_indices_helper(subtree.right_child)
                subtree.end_token = subtree.right_child.end_token

            else:     
                subtree.start_token = token_index[0]
                subtree.end_token = token_index[0] + len(subtree.tokens)
                token_index[0] += len(subtree.tokens)

        _assign_indices_helper(self.tree)

    def get_span_map(self, edus=None):
        span_map = {}

        def get_span_helper(subtree):
            if not subtree.is_leaf:
                span_map[(subtree.start_token, subtree.end_token)] = {
                    "nuclearity": subtree.nuclearity,
                    "relation": subtree.relation,
#                     "height": subtree.span.height_from_leaf,
#                     "depth": subtree.span.depth_from_root
                }
                get_span_helper(subtree.left_child)
                get_span_helper(subtree.right_child)
            else:
                span_map[(subtree.start_token, subtree.end_token)] = (
                    "Leaf",
                    subtree.tokens,
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
        print(identifier)
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
            
            nuc_match = rel1['nuclearity'] == rel2['nuclearity']
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

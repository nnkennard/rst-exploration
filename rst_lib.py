import json
import collections
import glob
import yaml




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


# def read_tag(parse):
#     if parse[0] == "span":
#         return {"span": tuple(int(x) for x in parse[1:])}
#     elif parse[0] == "text":
#         assert parse[1].startswith("_!") and parse[-1].endswith("_!")
#         tokens = [str(i) for i in parse[1:]]
#         tokens[0] = tokens[0][2:]
#         tokens[-1] = tokens[-1][:-2]
#         return {"text": " ".join(tokens), "tokens": tokens}
#     elif parse[0] == "leaf":
#         return {"leaf": int(parse[1])}
#     else:
#         assert parse[0] == "rel2par"
#         return {parse[0]: parse[1]}


def is_tag(parse):
    return all(type(x) in [str, float, int] for x in parse)


# def get_tags(parse):
#     tags = {}
#     for maybe_tag in parse:
#         if is_tag(maybe_tag):
#             tags.update(read_tag(maybe_tag))
#     return tags

def get_boundaries(text):
    index = 0
    paragraph_map = []
    sentence_map = []
    
    paragraphs = text.split("\n\n")
    sentence_start_index = 0
    for paragraph in paragraphs:
        paragraph_start = index
        sentences = paragraph.split("\n")
        for sentence in sentences:
            tokens = sentence.split()
            exclusive_end = index + len(tokens)
            sentence_map.append((index, exclusive_end))
            index = exclusive_end
        paragraph_map.append((paragraph_start, index))
    
    return {
        "paragraph": paragraph_map,
        "sentence": sentence_map
    }
    
    
with open('label_classes.yaml', 'r') as f:
    LABEL_CLASS_MAP = yaml.safe_load(f)
    
def get_label_class(label):
    if label[-4:] in ['-n-e', '-s-e']:
        return LABEL_CLASS_MAP[label[:-4]]
    elif label[-2:] in ['-n', '-e', 's']:
        return LABEL_CLASS_MAP[label[:-2]]
    else:
        return LABEL_CLASS_MAP[label]
        
        

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
    pairs = sum(
        [
            build_annotation_pair(files, paths, identifier)
            for identifier in files[TRAIN][DOUBLE]
        ],
        [],
    )
    if valid_only:
        return [pair for pair in pairs if pair.is_valid]
    else:
        return pairs


# =========== Objects for parse representation =============================

# Levels

INTRA_SENT, INTRA_PARA, INTER_PARA = "intra_sent intra_para inter_para".split()


class Subtree(object):
    def __init__(self, parse, binarizing=False):
        self._apply_tags(parse[1:])

        self.nuclearity_contrib = parse[0][0]
        assert self.nuclearity_contrib in "NSR"

        # Build children
        children = [subtree for subtree in parse if not is_tag(subtree)]
        self.is_leaf = len(children) == 0

        if not self.is_leaf:

            # Build tree
            self.left_child = Subtree(children[0])
            if len(children) == 2:
                self.right_child = Subtree(children[1])
            else:
                # If n-ary, build a right-branching binary tree
                self.right_child = Subtree(
                    ["Nucleus", ["rel2par", "span"]] + children[1:], binarizing=True
                )

            # Assign nuclearity
            if binarizing:
                self.nuclearity = "NN"  # Override: sometimes there is a S in a multinuclear relation (??)
            else:
                self.nuclearity = f"{self.left_child.nuclearity_contrib}{self.right_child.nuclearity_contrib}"

            # Assign relation
            self.relation, self.coarse_relation = self.determine_relation()

    def _apply_tags(self, parse):
        for maybe_tag in parse:
            if not is_tag(maybe_tag):
                continue
            if maybe_tag[0] == "span":
                continue  # We assign spans after binarization
            elif maybe_tag[0] == "leaf":
                self.start_edu = self.inc_end_edu = int(maybe_tag[1])
            elif maybe_tag[0] == "text":
                assert maybe_tag[1].startswith("_!") and maybe_tag[-1].endswith("_!")
                self.tokens = [str(i).replace("_!", "") for i in maybe_tag[1:]]
            else:
                assert maybe_tag[0] == "rel2par"
                self.rel2par = maybe_tag[1]

    def determine_relation(self):
        l = self.left_child.rel2par
        r = self.right_child.rel2par
        rel = None
        assert not l == r == "span"
        if not l == "span" and not r == "span":
            assert l == r
            rel = l
        elif l == "span":
            rel = r
        else:
            rel = l
            
        return rel, get_label_class(rel)


class Parse(object):
    def __init__(self, parse_text, boundaries):
        self.complete = False
        maybe_converted = bracket_conversion(parse_text)
        self.is_valid = maybe_converted is not None
        if self.is_valid:
            self.boundaries = boundaries
            parsed = parse(maybe_converted)
            self.tree = Subtree(parsed)
            self.edus, self.edu_indices = self._read_in_order()
            self._assign_span_indices()
            self.complete = True

    def _read_in_order(self):
        edus = [None] # No EDU 0

        def inorder_helper(tree):
            if tree.is_leaf:
                edus.append(tree.tokens)
            else:
                inorder_helper(tree.left_child)
                inorder_helper(tree.right_child)

        inorder_helper(self.tree)
        
        index = 0
        edu_indices = [None] # No EDU 0
        
        for edu in edus[1:]:
            next_index = index + len(edu)
            edu_indices.append((index, next_index))
            index = next_index
        
        return edus, edu_indices

    def _assign_span_indices(self):

        def _assign_indices_helper(subtree):
            # if subtree.is_weird_binary or not subtree.is_leaf:
            if not subtree.is_leaf:
                _assign_indices_helper(subtree.left_child)
                subtree.start_edu = subtree.left_child.start_edu
                
                _assign_indices_helper(subtree.right_child)
                subtree.inc_end_edu = subtree.right_child.inc_end_edu
            subtree.start_token, subtree.end_token, subtree.level = self._get_token_boundaries(subtree.start_edu, subtree.inc_end_edu)

                
        _assign_indices_helper(self.tree)
        
    def _get_token_boundaries(self, start_edu, inc_end_edu):
        start_token = self.edu_indices[start_edu][0]
        end_token = self.edu_indices[inc_end_edu][1]
        level = None
        
        same_para_found = False
        for paragraph_start, paragraph_end in self.boundaries['paragraph']:
            if start_token >= paragraph_start: # Start token is in this paragraph
                if end_token <= paragraph_end: # They are in the same paragraph; check for same sentence
                    
                    same_para_found = True
                    same_sentence_found = False
                    for sentence_start, sentence_end in self.boundaries['sentence']:
                        if start_token >= sentence_start:
                            if end_token <= sentence_end:
                                level = INTRA_SENT
                                same_sentence_found = True
                                
                    if not same_sentence_found:
                        level = INTRA_PARA
                    
        if not same_para_found:
            level = INTER_PARA
        
        return start_token, end_token, level

    def get_span_map(self, edus=None):
        span_map = {}

        def get_span_helper(subtree):
            if not subtree.is_leaf:
                span_map[(subtree.start_token, subtree.end_token)] = {
                    "nuclearity": subtree.nuclearity,
                    "relation": subtree.coarse_relation,
                    "level": subtree.level,
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
    def __init__(
        self,
        identifier,
        input_text,
        gold_annotation,
        predicted_annotation,
        main_is_gold,
    ):
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
            self.predicted_span_map = self.predicted_annotation.get_span_map(
                self.final_edus
            )
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
            "N": set(),
            "R": set(),
            "F": set(),
        }
        for span in matched_spans:

            rel1 = self.gold_span_map[span]
            rel2 = self.predicted_span_map[span]
            if type(rel1) == tuple or type(rel2) == tuple:
                continue

            nuc_match = rel1["nuclearity"] == rel2["nuclearity"]
            rel_match = rel1["relation"] == rel2["relation"]

            if nuc_match and rel_match:
                correct_spans["F"].add(span)
            if nuc_match:
                correct_spans["N"].add(span)
            if rel_match:
                correct_spans["R"].add(span)

        for k, v in correct_spans.items():
            f1_scores[k] = 2 * len(v) / jk

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
    boundaries = get_boundaries(main_in)
    if None in [main_out, double_out]:
        return
    return [
        AnnotationPair(
            identifier,
            main_in,
            Parse(main_out, boundaries),
            Parse(double_out, boundaries),
            main_is_gold=True
        ),
        AnnotationPair(
            f"{identifier}_r",
            main_in,
            Parse(double_out, boundaries),
            Parse(main_out, boundaries),
            main_is_gold=False,
        ),
    ]

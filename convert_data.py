import collections
import glob
import json
import lisp_lib
from transformers import RobertaTokenizer

TRAINING_PATH = "rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/*.dis"

ROBERTA_TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")


def bracket_conversion(parse_text):
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


def get_segmentation_format(parsed, edus):
    all_tokens = " ".join(sum(edus[1:], []))
    roberta_tokenized = ROBERTA_TOKENIZER.tokenize(all_tokens)
    labeled_roberta_tokens = []
    for edu in edus[1:]:
        first_added = False
        nonspace_chars_to_match = len("".join(edu))
        while nonspace_chars_to_match:
            curr_roberta_token = roberta_tokenized.pop(0)
            nonspace_chars_to_match -= len(curr_roberta_token)
            if curr_roberta_token.startswith("Ġ"):
                nonspace_chars_to_match += 1
            if first_added:
                label = "I"
            else:
                label = "B"
                first_added = True
            labeled_roberta_tokens.append((curr_roberta_token, label))
    return labeled_roberta_tokens


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


Span = collections.namedtuple("Span", "start end".split())
Relation = collections.namedtuple("Relation", "span_1 span_2 relation")
TextRelation = collections.namedtuple("TextRelation", "text_1 text_2 relation")


def get_span(tree):
    maybe_span = tree[1]
    if maybe_span[0] == "span":
        return Span(maybe_span[1], maybe_span[2])
    else:
        assert maybe_span[0] == "leaf"
        return Span(maybe_span[1], maybe_span[1])


def get_text_of_relation(relation, edus):
    span_tokens = []
    for span in [relation.span_1, relation.span_2]:
        start, incl_end = span
        tokens = []
        for i in range(start, incl_end + 1):
            tokens += edus[i]
        span_tokens.append(tokens)

    span_tokens_1, span_tokens_2 = span_tokens
    assert len(span_tokens) == 2
    return TextRelation(
        " ".join(span_tokens_1), " ".join(span_tokens_2), relation.relation
    )


def get_relation_format(parsed, edus):
    def get_relation_from_tree(tree):
        if len(tree) in [4, 5]:
            if len(tree) == 5:
              a, b, _, c, d = tree
            else:
              a, b, c, d = tree
            blerp = b[0]
            if blerp == "span":
                if c[0] == "Nucleus" and d[0] == "Satellite":
                    relation_name = d[2][1]
                elif d[0] == "Nucleus" and c[0] == "Satellite":
                    relation_name = c[2][1] + "_r"
                else:
                    assert d[0] == c[0] == "Nucleus"
                    relation_name = "same_unit"
                relations.append(Relation(get_span(c), get_span(d), relation_name))
            else:
                assert tree[0] == "text" or blerp == "leaf"

        for i, x in enumerate(tree):
            if type(x) not in [str, int, float]:
                get_relation_from_tree(x)

    relations = []
    get_relation_from_tree(parsed)
    return [get_text_of_relation(i, edus)._asdict() for i in relations]



def main():
    file_count = 0
    for filename in sorted(glob.glob(TRAINING_PATH)):
        with open(filename, "r") as f:
            text = f.read()
            new_text = bracket_conversion(text)
            if 'List' not in new_text:
              parse = lisp_lib.parse(new_text)
              edus = get_edu_list(parse)
              with open(f'rst_labeled_{file_count}.json', 'w') as g:
                json.dump(
              {
                "idx": file_count,
                "original_filename":filename,
                "segmentation": get_segmentation_format(parse, edus),
                "relation":get_relation_format(parse, edus)
              },
                g)
                file_count += 1


if __name__ == "__main__":
    main()

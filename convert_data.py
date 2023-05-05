import collections
import glob
import json
import lisp_lib
#from transformers import RobertaTokenizer

TRAINING_PATH = "rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/*.dis"

#ROBERTA_TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")

HORRIBLE_MAP = {
'antithesis': 'contrast',
'antithesis-e': 'contrast',
'attribution': 'attribution',
'attribution-e': 'attribution',
'attribution-n': 'attribution',
'attribution': 'attribution',
'background': 'background',
'circumstance': 'background',
'circumstance-e': 'background',
'comment': 'topic-comment',
'comment-e': 'topic-comment',
'comment': 'topic-comment',
'comparison': 'comparison',
'concession': 'contrast',
'concession-e': 'contrast',
'condition': 'condition',
'condition-e': 'condition',
'consequence-n': 'cause',
'consequence-n-e': 'cause',
'consequence-s': 'cause',
'consequence-s-e': 'cause',
'contingency': 'condition',
'elaboration-additional': 'elaboration',
'elaboration-additional-e': 'elaboration',
'elaboration-general-specific': 'elaboration',
'elaboration-general-specific-e': 'elaboration',
'elaboration-object-attribute': 'elaboration',
'elaboration-object-attribute-e': 'elaboration',
'elaboration-process-step': 'elaboration',
'elaboration-set-member': 'elaboration',
'elaboration-set-member-e': 'elaboration',
'enablement': 'enablement',
'evaluation-s': 'evaluation',
'evidence': 'explanation',
'example': 'elaboration',
'explanation-argumentative': 'explanation',
'hypothetical': 'condition',
'interpretation-s': 'evaluation',
'manner': 'mannermeans',
'means': 'mannermeans',
'means-e': 'mannermeans',
'otherwise': 'condition',
'purpose': 'enablement',
'purpose-e': 'enablement',
'reason': 'explanation',
'restatement': 'summary',
'restatement-e': 'summary',
'result': 'cause',
'rhetorical-question': 'topic-comment',
'same_unit': '',
'summary-n': 'summary',
'summary-s': 'summary',
'temporal-after': 'temporal',
'temporal-before': 'temporal',
'temporal-same-time': 'temporal',
}

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

  examples = []
  for edu in edus[1:]:
    begin_tok = edu[0]
    examples.append((begin_tok, 1))
    examples += [(tok, 0) for tok in edu[1:]]
  return examples


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
                    relation_name = HORRIBLE_MAP[d[2][1]]
                elif d[0] == "Nucleus" and c[0] == "Satellite":
                    relation_name = HORRIBLE_MAP[c[2][1]]
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
    segmentation_examples = {'dev':[], 'train':[]}
    for filename in sorted(glob.glob(TRAINING_PATH)):
        with open(filename, "r") as f:
            text = f.read()
            new_text = bracket_conversion(text)
            if 'List' not in new_text:
              parse = lisp_lib.parse(new_text)
              edus = get_edu_list(parse)
              tokens, labels = zip(*get_segmentation_format(parse, edus))
              if file_count % 10 == 0:
                subset = 'dev'
              else:
                subset = 'train'
              for i, chunk_start in enumerate(range(0, len(tokens), 200)):
                if chunk_start == 0:
                  offset = 0
                else:
                  offset = 20
                segmentation_examples[subset].append(
                {
                  f"original_filename_": filename,
                  "batch": i,
                  "tokens": tokens[chunk_start:chunk_start+200],
                  "labels": labels[chunk_start:chunk_start+200],}
                )
                file_count += 1
              #with open(f'relabeled/rst_labeled_{file_count}.json', 'w') as g:
              #  json.dump(
              #{
              #  "idx": file_count,
              #  "original_filename":filename,
              #  "segmentation": get_segmentation_format(parse, edus),
              #  "relation":get_relation_format(parse, edus)
              #},
              #  g)
              #  file_count += 1

    for subset, examples in segmentation_examples.items():
      with open(f'segmentation_data_{subset}.json', 'w') as f:
        json.dump({'data':examples},f, indent=4)



if __name__ == "__main__":
    main()

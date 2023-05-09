import json
import lisp_lib

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




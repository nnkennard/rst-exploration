"""
The rs3 format can be visualized at https://gucorpling.org/rstweb/structure.py.
Reference code for reading rs3 is https://github.com/amir-zeldes/rstWeb/blob/master/modules/rstweb_reader.py.
"""

import collections, json, os, re, yaml
import codecs
from xml.dom import minidom
import rst_lib

LABEL_CLASS_PRED = yaml.safe_load(open('label_classes_pred.yaml', 'r'))


class NodeFromRS3():
    def __init__(
            self, type, id, parent_id, relname, start_edu, inc_end_edu,
            children=None, nuclearity=None, relation=None, coarse_relation=None
    ):
        """
        Basic class to hold all nodes:
         - edu
         - span: hierarchical parent of a nucleus-satellite pair where the nucleus has a span relname
         - multinucs: hierarchical parent of multiple nuclei that have the same relname
        """

        self.type = type  # edu/span/multinuc
        self.start_edu = start_edu  # leftmost edu's index, may not be the same as the edu's rs3 id
        self.inc_end_edu = inc_end_edu  # rightmost edu's index, may not be the same as the edu's rs3 id

        # rs3 representation: https://gucorpling.org/rstweb/structure.py
        self.id = id
        self.parent_id = parent_id  # rs3 parent's id
        self.relname = relname  # rs3 relation

        # hierarchical children
        self.children = [] if children is None else children

        # binary tree representation
        # relation and nuclearity at hierarchical parents
        self.left_child = None  # hierarchical left child node
        self.right_child = None  # hierarchical right child node
        self.nuclearity = nuclearity  # hierarchical children's nuclearity (None for edu)
        self.relation = relation  # hierarchical children's relation
        self.coarse_relation = coarse_relation # hierarchical children's relation mapped by LABEL_CLASS_PRED


class ParseFromRS3():
    def __init__(self, identifier, file):
        self.identifier = identifier
        self.n_nodes_for_binarization = 0

        # Read the file
        data = codecs.open(file, "r", "utf-8").read()
        data = minidom.parseString(codecs.encode(data, "utf-8"))

        # Loop over relations that happen in this file.
        # Create disambiguated relation names with _r for rst and _m for multinuc.
        self.disambiguated_relnames = set()
        for row in data.getElementsByTagName("rel"):
            assert row.hasAttribute("type"), "There's a relation without a type."
            relname = row.attributes["name"].value
            assert relname == re.sub(r"[:;,]", "", relname)
            
            type = row.attributes["type"].value
            self.disambiguated_relnames.add(f"{relname}_{type[0:1]}")

        # Loop over segments (edu).
        self.id_to_type = {}  # rs3 id -> edu/span/multinuc
        self.edu_id_to_idx = {}  # edu indexes to be later used for "start_edu" and "inc_end_edu"

        rows = data.getElementsByTagName("segment")
        assert len(rows) >= 1, f"No segment found in {file}."
        for i, row in enumerate(rows):
            assert row.hasAttribute("parent") and row.hasAttribute("relname")
            self.id_to_type[row.attributes["id"].value] = "edu"

            # assume the segment order reflects the edu order in the original text
            self.edu_id_to_idx[row.attributes["id"].value] = i + 1

        # Loop over groups (span, multinuc).
        has_root = False  # check if there's only one root
        for row in data.getElementsByTagName("group"):
            id = row.attributes["id"].value
            type = row.attributes["type"].value
            assert type in ["span", "multinuc"], f"Group type is neither span nor multinuc: {type}."
            self.id_to_type[id] = type

            if row.hasAttribute("parent"):
                assert row.hasAttribute("relname")
            else:  # must be the root
                assert not has_root, f"two potential roots in {file}"
                self.root_id, has_root = id, True
        assert has_root, f"No root in {file}."
        
        # Create nodes!
        self.nodes = {}
        rows = data.getElementsByTagName("segment") + data.getElementsByTagName("group")
        for row in rows:
            id = row.attributes["id"].value
            type = self.id_to_type[id]  # edu/span/multinuc
            self.nodes[id] = NodeFromRS3(
                type=type,
                id=id,
                parent_id=row.attributes["parent"].value if row.hasAttribute("parent") else None,
                relname=row.attributes["relname"].value if row.hasAttribute("parent") else None,
                start_edu=self.edu_id_to_idx[id] if type == "edu" else None,
                inc_end_edu=self.edu_id_to_idx[id] if type == "edu" else None,
            )

        # Loop over segments (edu) and groups (span, multinuc) to figure out
        # the multinuc relation among each multinuc parent node's children.
        multinuc_relname_children = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # multinuc_relname_children[multinuc_parent_id][relname] = count
        for row in rows:
            if not row.hasAttribute("parent"):  # the root has no parent
                continue

            id = row.attributes["id"].value
            parent_id = row.attributes["parent"].value
            assert parent_id in self.id_to_type, f"The rs3 parent of a node is not another node in {file}."
            if self.id_to_type[parent_id] == "multinuc":  # parent is a multinuc
                multinuc_relname_children[parent_id][row.attributes["relname"].value].append(id)

        multinuc_to_children = {}
        self.multinuc_to_reshape = set()
        for parent_id, relname_children in multinuc_relname_children.items():
            assert len(relname_children) in [1, 2], \
                "A multinuc parent's rs3 children should have one or two relations."

            # If a multinuc parent's children have one relation, there are two possibilities.
            if len(relname_children) == 1:
                relname = list(relname_children.keys())[0]
                # fake multinuc: just one child that has a span relation
                if relname == "span":
                    assert len(relname_children[relname]) == 1
                    continue
                assert f"{relname}_m" in self.disambiguated_relnames
                children = relname_children[relname]

                if self.nodes[parent_id].relname == "span":
                    # (1) Let the leftmost child be a sibling satellite and others hierarchical children.
                    #     We don't know which child is the leftmost and thus
                    #     keep all children for now and do the reshaping later.
                    assert len(children) >= 3
                    self.multinuc_to_reshape.add(parent_id)
                    multinuc_to_children[parent_id] = children
                else:
                    # (2) All are hierarchical children.
                    multinuc_to_children[parent_id] = children
            # If a multinuc parent's children have two relations,
            # one child is a sibling satellite and others are hierarchical children.
            else:
                relnames = list(relname_children.keys())
                if len(relname_children[relnames[0]]) > 1:
                    multinuc_relname = relnames[0]
                    assert len(relname_children[relnames[1]]) == 1
                else:
                    multinuc_relname = relnames[1]
                    assert len(relname_children[relnames[1]]) > 1
                assert f"{multinuc_relname}_m" in self.disambiguated_relnames
                multinuc_to_children[parent_id] = relname_children[multinuc_relname]

        # Connect each node to its hierarchical parent; add the relation to the parent
        for node_id, node in self.nodes.items():
            # the root has no parent
            if node.parent_id is None:
                assert node_id == self.root_id
                continue

            parent = self.nodes[node.parent_id]
            # this node has a span relation with its hierarchical parent and is the nucleus of its sibling
            if node.relname == "span":
                parent.children.append(node)
            # this node has a hierarchical multinuc parent and a multinuc relation with its sibling(s)
            elif parent.type == "multinuc" and node.id in multinuc_to_children[node.parent_id]:
                parent.children.append(node)
                if parent.relation is None:
                    parent.relation = node.relname
                    parent.coarse_relation = LABEL_CLASS_PRED[parent.relation]
                else:
                    assert parent.relation == node.relname
            else:
                # the parent is a sibling nucleus, this node is the satellite
                # nodes that satisfy this and the above conditions are handled as the above condition
                assert parent.relname == "span"
                grandparent = self.nodes[parent.parent_id]
                grandparent.children.append(node)
                assert grandparent.relation is None
                grandparent.relation = node.relname
                grandparent.coarse_relation = LABEL_CLASS_PRED[grandparent.relation]

        # Cleaning and building the binary tree
        n_processed_nodes = self._assign_start_end_edus(self.nodes[self.root_id])
        assert n_processed_nodes == len(rows), \
            f"Reached {n_processed_nodes} nodes when traversing the hierarchy from the root, " \
            f"but there should be {len(rows)} nodes."
        self._normalize_nodes(self.nodes[self.root_id])
        self._add_nuclearity(self.nodes[self.root_id])
        self._binarization(self.nodes[self.root_id])

    def _assign_start_end_edus(self, node, n_processed_nodes=0):
        """For each node, add start_edu and inc_end_edu"""
        if node.type != "edu":
            for child in node.children:
                n_processed_nodes = self._assign_start_end_edus(child, n_processed_nodes)
            node.start_edu = min([child.start_edu for child in node.children])
            node.inc_end_edu = max([child.inc_end_edu for child in node.children])
        return n_processed_nodes + 1

    def _normalize_nodes(self, node):
        """
        For each node, order its children based on start_edu.
        Fix multinuc nodes with a span relation and one rs3 children relation.
        """
        if node.type != "edu":
            node.children.sort(key=lambda child: child.start_edu)
            assert node.start_edu == node.children[0].start_edu
            assert node.inc_end_edu == node.children[-1].inc_end_edu
            for i in range(len(node.children) - 1):
                assert node.children[i].inc_end_edu + 1 == node.children[i + 1].start_edu

            # reshape if a node is a multinuc, has a span relation, and
            # one children relation (i.e. no currently sibling satellite)
            if node.id in self.multinuc_to_reshape:
                assert len(node.children) > 2
                satellite = node.children[0]
                node.children = node.children[1:]
                node.start_edu = node.children[0].start_edu

                parent = self.nodes[node.parent_id]
                assert len(parent.children) == 1 and parent.children[0] == node
                parent.children = [satellite, node]
                assert parent.relation is None
                parent.relation = node.relation
                parent.coarse_relation = node.coarse_relation

                self._normalize_nodes(satellite)

            for child in node.children:
                self._normalize_nodes(child)

    def _add_nuclearity(self, node):
        if node.type != "edu":
            n_children = len(node.children)
            assert n_children >= 2
            if len(node.children) == 2:
                if node.children[0].relname == "span":
                    node.nuclearity = "NS"
                elif node.children[1].relname == "span":
                    node.nuclearity = "SN"
                else:
                    assert node.children[0].relname == node.children[1].relname
                    node.nuclearity = "NN"
            else:
                node.nuclearity = "N" * n_children

            for child in node.children:
                self._add_nuclearity(child)

    def _binarization(self, node):
        """
        Binarize the tree.
        For each node, specify its left_child and right_child; verify that the relation is added.
        """
        if node.type != "edu":
            assert node.relation is not None and node.coarse_relation is not None

            node.left_child = node.children[0]
            if len(node.children) == 2:
                node.right_child = node.children[1]
            else:
                for c in node.children:
                    assert node.relation == c.relname, "Multinuc children should not have different relations."
                node.nuclearity = "NN"

                self.n_nodes_for_binarization += 1
                id = f"b{self.n_nodes_for_binarization}"
                node.right_child = NodeFromRS3(
                    type="multinuc",
                    id=id,  # pseudo rs3 id
                    parent_id=None,  # no rs3 id
                    relname=None,  # no rs3 id
                    start_edu=node.children[1].start_edu,
                    inc_end_edu=node.children[-1].inc_end_edu,
                    children=node.children[1:],
                    nuclearity="N" * len(node.children[1:]),
                    relation=node.relation,
                    coarse_relation=node.coarse_relation,
                )
                self.nodes[id] = node.right_child
                self.id_to_type[id] = "multinuc"

            self._binarization(node.left_child)
            self._binarization(node.right_child)

    def get_spans(self):
        spans = []

        def get_span(node):
            if node.type != "edu":
                spans.append({
                    "identifier": self.identifier,
                    "start_edu": node.start_edu,
                    "end_edu": node.inc_end_edu,
                    "nuclearity": node.nuclearity,
                    "relation": node.relation,
                    "coarse_relation": node.coarse_relation
                })
                get_span(node.left_child)
                get_span(node.right_child)

        get_span(self.nodes[self.root_id])
        return spans


if __name__ == '__main__':
    PARSERS_PRED_DIR = "/work/pi_mccallum_umass_edu/wenlongzhao_umass_edu/RST/Janet-test-predictions/"
    PARSERS = ["bottomup", "topdown"]
    RUNS = ["RUN1", "RUN2", "RUN3", "RUN4", "RUN5"]

    _, files = rst_lib.build_file_map("./rst_discourse_treebank/")
    identifiers = sorted(sum(files['test'].values(), []))

    label_dicts = collections.defaultdict(lambda: collections.defaultdict(list))
    for parser in PARSERS:
        for run in RUNS:
            print(parser, run)
            experiment_dir = os.path.join(
                PARSERS_PRED_DIR, f"RSTDT_{parser}/{run}/predicted_trees_rs3"
            )

            # loop over test samples to get predictions by a parser ckpt
            for identifier in identifiers:
                file = os.path.join(experiment_dir, f"{identifier}.rs3")

                data = codecs.open(file, "r", "utf-8").read()
                data = minidom.parseString(codecs.encode(data, "utf-8"))
                parse = ParseFromRS3(identifier, file)
                label_dicts[parser][run].extend(parse.get_spans())
    os.makedirs("data", exist_ok=True)
    with open(f"data/converted_parser_predictions.json", 'w') as f:
        json.dump(label_dicts, f, indent=4)
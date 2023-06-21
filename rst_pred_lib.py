"""
Referred to https://github.com/amir-zeldes/rstWeb/blob/master/modules/rstweb_reader.py
"""

import codecs, collections, re, yaml
from xml.dom import minidom


with open('label_classes_pred.yaml', 'r') as f:
    LABEL_CLASS_PRED = yaml.safe_load(f)


class NodeFromRS3():
    def __init__(self, type, parent, relname, start_edu, inc_end_edu):
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
        self.parent = parent  # rs3 parent's id
        self.relname = relname  # rs3 relation

        # binary tree representation, relation and nuclearity at hierarchical parents
        self.left_child = None  # hierarchical left child's rs3 id
        self.right_child = None  # hierarchical right child's rs3 id
        self.nuclearity = None  # hierarchical children's nuclearity (None for edu)
        self.relation = None  # hierarchical children's relation


class ParseFromRS3():
    def __init__(self, identifier, file):
        self.identifier = identifier

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
                self.root, has_root = id, True
        assert has_root, f"No root in {file}."

        # Loop over segments (edu) and groups (span, multinuc) to populate multinuc_relname_count,
        # figuring out the multinuc relation among each multinuc parent node's children.
        multinuc_relname_count = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )  # multinuc_relname_count[multinuc_parent_id][relname] = count
        multinuc_to_relname = {}
        rows = data.getElementsByTagName("segment") + data.getElementsByTagName("group")
        for row in rows:
            if not row.hasAttribute("parent"):  # the root has no parent
                continue
            parent = row.attributes["parent"].value
            relname = row.attributes["relname"].value
            assert parent in self.id_to_type, f"The rs3 parent of a node is not another node in {file}."
            # In this case, the relation could be a multinuc relation.
            if self.id_to_type[parent] == "multinuc":
                multinuc_relname_count[parent][relname] += 1

        for parent, relname_count in multinuc_relname_count.items():
            n_relnames = len(relname_count)
            assert n_relnames in [1, 2], \
                "A multinuc parent's rs3 children should have one or two relations."

            # If a multinuc parent's children have two relations,
            # one child is a sibling satellite and others are hierarchical children.
            for relname in multinuc_relname_count[parent]:
                if multinuc_relname_count[parent][relname] > 1:
                    assert parent not in multinuc_to_relname, \
                        "Multiple multinuc relations are identified for a multinuc parent."
                    multinuc_to_relname[parent] = relname
            assert multinuc_to_relname[parent] + "_m" in self.disambiguated_relnames, \
                "An identified multinuc relation is not a possible relation."

        # Now create nodes!
        self.nodes = {}
        for row in rows:
            id = row.attributes["id"].value
            type = self.id_to_type[id]  # edu/span/multinuc
            self.nodes[id] = NodeFromRS3(
                type=type,
                parent=row.attributes["parent"].value if row.hasAttribute("parent") else None,
                relname=row.attributes["relname"].value if row.hasAttribute("parent") else None,
                start_edu=self.edu_id_to_idx[id] if type == "edu" else None,
                inc_end_edu=self.edu_id_to_idx[id] if type == "edu" else None,
            )

        print("OMG, success!")
        dd

        # # Connect each node to its hierarchical parent;
        # # add the relation and the nuclearity to the parent
        # for node in self.nodes.values():
        #     # the root has no parent
        #     if node.parent is None:
        #         continue
        #     # the parent can be a sibling nucleus, this node is the satellite
        #     elif node.parent
        #
        #     # the parent is a hierarchical parent, this node has a span relation and is the nucleus of its sibling
        #     elif
        #
        #     # the parent is a hierarchical multinuc parent, this node is one of the children in the multinuc relation
        #
        #     self.nodes[node.parent].children.append(node)
        #
        # # Transform the hierarchy to be a binary tree;
        # # specify left_child and right_child and update relation and nuclearity for each node
        # n_added_nodes = 0
        #
        # # Populate start_edu and inc_end_edu
        # n_processed_nodes = self._assign_start_end_edus(self.root)
        # assert n_processed_nodes == len(rows) + n_added_nodes, \
        #     f"n_processed_nodes {n_processed_nodes} != n_nodes {len(rows)} + n_added_nodes {n_added_nodes}"

    def _assign_start_end_edus(self, node, n_processed_nodes=0):
        if node.type != "edu":
            n_processed_nodes = _assign_start_end_edus(node.left_child, n_processed_nodes)
            n_processed_nodes = _assign_start_end_edus(node.right_child, n_processed_nodes)
            node.start_edu = node.left_child.start_edu
            node.inc_end_edu = node.right_child.inc_end_edu
        return n_processed_nodes + 1

    def get_spans(self):
        spans = []

        def get_span(node):
            if not node.is_leaf:
                spans.append({
                    "identifier": self.identifier,
                    "start_edu": node.start_edu,
                    "end_edu": node.inc_end_edu,
                    "nuclearity": node.nuclearity,
                    "orig_relation": node.relation,
                })
                get_span(node.left_child)
                get_span(node.right_child)

        get_span(self.root)
        return span_map


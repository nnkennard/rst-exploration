import collections
import glob
from rst_lib import Parse
from rst_lib import build_file_map, build_annotation_pair, TRAIN, DOUBLE
import yaml


Metrics = collections.namedtuple("Metrics", "s_p s_r r_p r_r")

with open("label_classes.yaml", "r") as f:
    LABEL_CLASS_MAP = yaml.load(f.read(), Loader=yaml.Loader)


def sort_by_length(spans):
    d = collections.defaultdict(list)
    for a, b in spans:
        d[b - a + 1].append((a, b))

    for l, v in d.items():
        print("span len ", l, "num spans", len(v))


def get_span_match_map(span_map_1, span_map_2):
    tp_spans = set(span_map_1.keys()).intersection(span_map_2.keys())
    fn_spans = set(span_map_1.keys()) - set(span_map_2.keys())
    fp_spans = set(span_map_2.keys()) - set(span_map_1.keys())

    assert (len(span_map_1) + len(span_map_2) - len(tp_spans)) == (
        len(fp_spans) + len(fn_spans) + len(tp_spans)
    )

    print("true positives")
    sort_by_length(tp_spans)
    print("false positives")
    sort_by_length(fp_spans)
    print("false negatives")
    sort_by_length(fn_spans)
    print()


def get_metrics(span_map_1, span_map_2):
    num_orig_spans = len(span_map_1)
    num_final_spans = len(span_map_2)
    true_positive_spans = set(span_map_1.keys()).intersection(span_map_2.keys())

    true_positive_relation_count = 0
    for span in true_positive_spans:
        if LABEL_CLASS_MAP[span_map_1[span]] == LABEL_CLASS_MAP[span_map_2[span]]:
            true_positive_relation_count += 1

    return Metrics(
        s_p=len(true_positive_spans) / num_final_spans,
        s_r=len(true_positive_spans) / num_orig_spans,
        r_p=true_positive_relation_count / num_final_spans,
        r_r=true_positive_relation_count / num_orig_spans,
    )


def get_flips(span_map_1, span_map_2):
    flips = []
    true_positive_spans = set(span_map_1.keys()).intersection(span_map_2.keys())
    for span in true_positive_spans:
        a = LABEL_CLASS_MAP[span_map_1[span]]
        b = LABEL_CLASS_MAP[span_map_2[span]]
        if not a == b:
            flips.append(tuple(list(sorted([a, b]))))

    return flips


def f1_score(p, r):
    if not p + r:
        return 0
    return 2 * p * r / (p + r)


def mean(l):
    return sum(l) / len(l)


def main():
    paths, files = build_file_map()

    annotation_pairs = [
        build_annotation_pair(files, paths, identifier)
        for identifier in files[TRAIN][DOUBLE]
    ]

    f1s = {
        "s": [],
        "r": [],
    }

    flips = []
    for x in annotation_pairs:
        if x is None or not x[3].is_valid or not x[2].is_valid:
            continue
        # print(x.identifier)
        main_span_map = x.main_annotation.span_map
        double_span_map = x.double_annotation.span_map
        # print(len(main_span_map), len(double_span_map))

        # match_map =get_span_match_map(main_span_map, double_span_map)

        # metrics = get_metrics(main_span_map, double_span_map)
        # print(x.identifier, f1_score(metrics.s_p, metrics.s_r))
        # f1s['s'].append(f1_score(metrics.s_p, metrics.s_r))
        # f1s['r'].append(f1_score(metrics.r_p, metrics.r_r))
        flips += get_flips(main_span_map, double_span_map)

    c = collections.Counter(flips)
    for k, v in c.most_common():
        print(k, v)
    # print(mean(f1s['s']))
    # print(mean(f1s['r']))


if __name__ == "__main__":
    main()

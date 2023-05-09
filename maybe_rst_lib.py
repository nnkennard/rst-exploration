import collections
import glob
from rst_parse import Parse


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


def get_file_pair(path, input_file):
    with open(f"{path}{input_file}", "r") as f:
        input_text = f.read()
    with open(f"{path}{input_file}.dis", "r") as f:
        output_text = f.read()
    return input_text, output_text


def build_annotation_pair(main_path, double_path, identifier):
    if identifier.startswith("file"):
        input_file = identifier
    else:
        input_file = f"{identifier}.out"
    main_in, main_out = get_file_pair(main_path, input_file)
    double_in, double_out = get_file_pair(double_path, input_file)
    assert main_in == double_in
    if None in [main_out, double_out]:
        return
    return AnnotationPair(main_in, Parse(main_out), Parse(double_out))


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

annotation_pairs = [
    build_annotation_pair(paths[TRAIN], paths[DOUBLE], identifier)
    for identifier in files[TRAIN][DOUBLE]
]

for x in annotation_pairs:
    if x is None or not x[1].is_valid or not x[2].is_valid:
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

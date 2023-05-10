import collections
import glob
from rst_parse import Parse

AnnotationPair = collections.namedtuple(
    "AnnotationPair", "identifier input_text main_annotation double_annotation"
)

TRAIN, TEST, DOUBLE, SINGLE = "train test double single".split()

DATA_PATH = "./rst_discourse_treebank/"


def get_file_pair(path, input_file):
    with open(f"{path}{input_file}", "r") as f:
        input_text = f.read()
    with open(f"{path}{input_file}.dis", "r") as f:
        output_text = f.read()
    return input_text, output_text


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
    return AnnotationPair(identifier, main_in, Parse(main_out), Parse(double_out))


def build_file_map(data_path=DATA_PATH):
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

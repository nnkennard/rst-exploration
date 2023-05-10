import collections
import glob
from rst_parse import Parse
from annotation_pairs import build_file_map, build_annotation_pair
from annotation_pairs import TRAIN, DOUBLE
import yaml


Metrics = collections.namedtuple("Metrics", "s_p s_r r_p r_r")

with open('label_classes.yaml', 'r') as f:
  LABEL_CLASS_MAP = yaml.load(f.read(), Loader=yaml.Loader)

def get_metrics(span_map_1, span_map_2):

  num_orig_spans = len(span_map_1)
  num_final_spans = len(span_map_2)
  true_positive_spans = set(span_map_1.keys()).intersection(span_map_2.keys())

  true_positive_relation_count = 0
  for span in true_positive_spans:
    if LABEL_CLASS_MAP[span_map_1[span]] == LABEL_CLASS_MAP[span_map_2[span]]:
      true_positive_relation_count += 1

  return Metrics(
    s_p=len(true_positive_spans)/num_final_spans,
    s_r=len(true_positive_spans)/num_orig_spans,
    r_p=true_positive_relation_count/num_final_spans,
    r_r=true_positive_relation_count/num_orig_spans)


def f1_score(p, r):
  if not p+r:
    return 0
  return 2* p * r/(p+r)

def mean(l):
  return sum(l)/len(l)

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

  for x in annotation_pairs:
      if x is None or not x[3].is_valid or not x[2].is_valid:
          continue
      print(x.identifier)
      main_span_map = x.main_annotation.span_map
      double_span_map = x.double_annotation.span_map
      metrics = get_metrics(main_span_map, double_span_map)
      f1s['s'].append(f1_score(metrics.s_p, metrics.s_r))
      f1s['r'].append(f1_score(metrics.r_p, metrics.r_r))

  print(mean(f1s['s']))
  print(mean(f1s['r']))



if __name__ == "__main__":
  main()

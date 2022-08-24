import argparse
import collections
import glob
import json
import openreview
import os
import pickle
import tqdm

import rst_lib

DisapereExample = collections.namedtuple(
    "DisapereExample", "review_id forum_id review_sentences rebuttal_sentences url")

GUEST_CLIENT = openreview.Client(baseurl="https://api.openreview.net")

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-p",
    "--disapere_path",
    default="/iesl/canvas/nnayak/DISAPERE/DISAPERE/final_dataset/",
    type=str,
    help="Path to DISAPERE final_dataset directory",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default="pdfs/",
    type=str,
    help="Path to DISAPERE pdf output directory",
)


def get_sentences(obj, sentence_type):
  assert sentence_type in ["review", "rebuttal"]
  return [x["text"] for x in obj[f"{sentence_type}_sentences"]]

def write_pdf(reference_id, first_or_last, forum_id, pdf_dir):
  filename = f'{pdf_dir}/{forum_id}_{first_or_last}.pdf'
  if os.path.exists(filename):
    # expected, multiple reviews per forum
    dsds
    print("Problem???", filename)
    return
  with open(filename, 'wb') as f:
    print(GUEST_CLIENT.get_pdf(reference_id, is_reference=True))
    f.write(GUEST_CLIENT.get_pdf(reference_id, is_reference=True))

def download_first_last(forum_id, pdf_dir):
  references = sorted(GUEST_CLIENT.get_references(forum_id, original=True),
                      key=lambda x: x.tcdate)
  print(forum_id)
  first = False
  for r in references:
    write_pdf(r, 'first', forum_id, pdf_dir)
    try:
      write_pdf(r, 'first', forum_id, pdf_dir)
      first = True
      print("Wrote first", r)
    except:
      continue
    if first:
      break
  for r in reversed(references):
    try:
      write_pdf(r, 'last', forum_id, pdf_dir)
    except:
      continue
  
def load_examples(disapere_path, pdf_dir, subset="train"):
  examples = []
  for filename in tqdm.tqdm(glob.glob(f"{disapere_path}/{subset}/*")):
    with open(filename, "r") as f:
      obj = json.load(f)
      examples.append(
          DisapereExample(
              obj["metadata"]["review_id"],
              obj["metadata"]["forum_id"],
              get_sentences(obj, "review"),
              get_sentences(obj, "rebuttal"),
              obj["metadata"]["permalink"],
          ))
      #download_first_last(obj["metadata"]["forum_id"], pdf_dir)
  return examples


def main():

  args = parser.parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  examples = load_examples(args.disapere_path, args.output_dir)
  with open('disapere_train_temp.pkl', 'wb') as f:
    pickle.dump([x._asdict() for x in examples], f)



if __name__ == "__main__":
  main()

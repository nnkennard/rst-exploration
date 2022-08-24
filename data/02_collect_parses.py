import collections
import glob
import json
import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer

DiscourseUnit = collections.namedtuple("DiscourseUnit",
    "forum_id part index sentence tokens embedding")

Triple = collections.namedtuple("Triple",
    "forum_id sections review rebuttal")

def cosine(v1, v2):
  return np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))

class Triple(object):
  def __init__(self, forum_id, sections, review, rebuttal):
    self.forum_id = forum_id
    self.sections = sections

  def get_embeddings_map(self, key):
    return {s.sentence: s.embedding for s in self.sections[key]}

def build_triple(forum_id, parts, model):

  sentence_map = collections.defaultdict(list)
  for part_id, part in parts.items():
    tokens = part['edu_tokens']
    sentences = [" ".join(x) for x in tokens]
    embeddings = model.encode(sentences)
    for i, (token, sentence, embedding) in enumerate(zip(tokens, sentences,
      embeddings)):
      sentence_map[part_id].append(DiscourseUnit(forum_id, part_id, i, sentence,
      tuple(token), embedding))

  return Triple(forum_id, sentence_map, None, None)


def main():

  text_map = {}
  for filename in glob.glob("rst_texts/input/*/*.txt"):
    _, _, forum_id, part = filename.split("/")
    part = part[:-4]
    with open(filename, 'r') as f:
      text = f.read()[:400]
      text_start = "".join(text.split())[:200]
      if not text_start:
        continue
      if text_start in text_map:
        continue
      assert text_start not in text_map
      text_map[text_start] = forum_id, part

  texts_and_parses = collections.defaultdict(dict)
  with open("output.json", 'r') as f:
    for line in f:
      obj = json.loads(line)
      maybe_key = "".join("".join(x) for x in obj['edu_tokens'])[:200]
      if maybe_key in text_map:
        forum_id, part = text_map[maybe_key]
        texts_and_parses[forum_id][part] = obj

  model = SentenceTransformer('all-MiniLM-L6-v2')
  triples = []
  for forum_id, parts in tqdm.tqdm(texts_and_parses.items()):
    if len(parts.keys()) < 3:
      continue
    maybe_numbers = [int(x.split("_")[1]) for x in parts.keys() if "_" in x]
    if list(sorted(maybe_numbers)) == list(range(len(maybe_numbers))):
      triples.append(build_triple(forum_id, parts, model))


  for triple in triples:
    review_embeddings = triple.get_embeddings_map("review")
    rebuttal_embeddings = triple.get_embeddings_map("rebuttal")

    for rebuttal_sentence, rebuttal_embedding in rebuttal_embeddings.items():
      max_cosine = -1.0
      cosine_list = [(review_sentence, cosine(rebuttal_embedding,
        review_embedding))
        for review_sentence, review_embedding in review_embeddings.items()]
      sorted_cosine_list = sorted(cosine_list, key=lambda x: -1 * x[1])

      print(rebuttal_sentence)
      for rev_sent, _ in sorted_cosine_list[:5]:
        print("   "+rev_sent)
      print()
    print("-" * 80)



if __name__ == "__main__":
  main()

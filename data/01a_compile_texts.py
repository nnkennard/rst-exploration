import collections
import os
import pickle
import xml.etree.ElementTree as ET

import stanza
STANZA_PIPELINE = stanza.Pipeline('en',
        processors='tokenize',
       )

PREFIX = "{http://www.tei-c.org/ns/1.0}"

DIV_ID = f"{PREFIX}div"
HEAD_ID = f"{PREFIX}head"
P_ID = f"{PREFIX}p"

def first_child(parent, child_part):
    return parent.findall(f'{PREFIX}{child_part}')[0]

def get_docs(filename):
  section_titles = []
  section_texts = []
  try:
    root = ET.parse(filename).getroot()
  except:
    return None
  divs = first_child(first_child(root, "text"), "body").findall(DIV_ID)
  for div in divs:
    head_nodes = div.findall(HEAD_ID)
    if not head_nodes:
      return None
    else:
      header_node = head_nodes[0]
    section_titles.append(header_node.text)
    text = ""
    for p in div.findall(P_ID):
      text += " ".join(p.itertext())
    section_texts.append(text)
  return list(zip(section_titles, section_texts))

class Section(object):
  def __init__(self, header, raw_text):
    self.header = header
    doc = STANZA_PIPELINE(raw_text)
    self.sentences = [sent.text for sent in doc.sentences]

class Forum(object):
  def __init__(self, examples):
    e = examples[0]
    self.forum_id = e['forum_id']
    maybe_docs = get_docs(f"xmls/{e['forum_id']}_first.tei.xml")
    if maybe_docs is None:
      self.is_valid = False
      return
    else:
      self.is_valid = True
      self.sections = [Section(h, t) for h, t in maybe_docs]
      self.reviews = [e["review_sentences"] for e in examples]
      self.rebuttals = [e["rebuttal_sentences"] for e in examples]

  def write_files(self, write_dir):
    os.makedirs(f'{write_dir}/{self.forum_id}/', exist_ok=True)
    if not self.is_valid:
      return
    for i, section in enumerate(self.sections):
      filename = f'{write_dir}/{self.forum_id}/section_{i}.txt'
      with open(filename, 'w') as f:
        f.write(" ".join(section.sentences))
    for i, review in enumerate(self.reviews):
      filename = f'{write_dir}/{self.forum_id}/review_{i}.txt'
      with open(filename, 'w') as f:
        f.write(" ".join(review))
    for i, rebuttal in enumerate(self.rebuttals):
      filename = f'{write_dir}/{self.forum_id}/rebuttal_{i}.txt'
      with open(filename, 'w') as f:
        f.write(" ".join(rebuttal))





def main():

  review_map = collections.defaultdict(list)

  with open('disapere_train_temp.pkl', 'rb') as f:
    for example in pickle.load(f):
      review_map[example["forum_id"]].append(example)

  forums = []
  for k, v in review_map.items():
    forums.append(Forum(v))

  for f in forums:
    f.write_files("prepared_texts/")

if __name__ == "__main__":
  main()

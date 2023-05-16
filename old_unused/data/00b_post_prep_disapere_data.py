import glob
import os
import pickle

import xml.etree.ElementTree as ET

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
    return None, None
  divs = first_child(first_child(root, "text"), "body").findall(DIV_ID)
  for div in divs:
    head_nodes = div.findall(HEAD_ID)
    if not head_nodes:
      return None, None
    else:
      header_node = head_nodes[0]
    section_titles.append(header_node.text)
    text = ""
    for p in div.findall(P_ID):
      text += " ".join(p.itertext())
    section_texts.append(text)
  return section_titles, section_texts

def main():
  with open("disapere_train_temp.pkl", 'rb') as f:
    examples = pickle.load(f)
    for i, e in enumerate(examples):
      headers, texts = get_docs(f"xmls/{e['forum_id']}_first.tei.xml")
      if headers is not None:
        path = f'rst_texts/input/{e["review_id"]}'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/review.txt', 'w') as g:
          g.write(" ".join(e['review_sentences']))
        with open(f'{path}/rebuttal.txt', 'w') as g:
          g.write(" ".join(e['rebuttal_sentences']))
        for j, text in enumerate(texts):
          with open(f'{path}/section_{j}.txt', 'w') as g:
            g.write(text)

  filenames = sorted(glob.glob("rst_texts/input/*/*"))
  print("\n".join(filenames))

if __name__ == "__main__":
  main()

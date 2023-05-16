import collections
import glob
import re

GUM_PATH = "/work/nnayak_umass_edu/gum/coref/gum/tsv/GUM_*" 

def main():

  with open('marcu-cue-phrases.txt', 'r') as f:
    cues = [line.strip() for line in f if not line.startswith("#")]


  sentences = collections.defaultdict(lambda:collections.defaultdict(list))
  cue_starts = collections.defaultdict(collections.Counter)
  cue_middles = collections.defaultdict(collections.Counter)

  for filename in glob.glob(GUM_PATH):
    category, title = filename[:-4].split("_")[-2:]
    with open(filename, 'r') as f:
      for line in f:
        if line.startswith("#Text="):
          sentences[category][title].append(line.strip()[6:].lower())

  for category, articles in sentences.items():
    for title, sentence_list in articles.items():
      for sentence in sentence_list:
        for cue in cues:
          if sentence.startswith(cue + " "):
            cue_starts[category][cue] += 1
          elif " " + cue + " " in sentence:
            cue_middles[category][cue] += len(re.findall(" " +  cue + " ", sentence))


  for category, cue_start_map in cue_starts.items():
    total_start_cues = sum(cue_start_map.values())
    total_middle_cues = sum(cue_middles[category].values())
    total_sentences = sum(len(x) for x in sentences[category].values())
    print(category, total_start_cues/total_sentences)

if __name__ == "__main__":
  main()


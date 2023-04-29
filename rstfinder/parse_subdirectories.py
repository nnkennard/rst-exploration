import glob
import subprocess
import tqdm


def main():
  for path in tqdm.tqdm(glob.glob("../data/prepared_texts/*")):
    forum = path.split("/")[-1]
    with open(f'{forum}.jsonl', 'w') as f:
      p = [
          "rst_parse",
          "-g",
          "segmentation_model.C1.0",
          "-p",
          "rst_parsing_model.C1.0",
          f'../data/prepared_texts/{forum}/*'
          
          ]
      subprocess.run(" ".join(p),stdout=f, shell=True   )


if __name__ == "__main__":
  main()

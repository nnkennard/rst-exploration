import collections
import glob
import subprocess

from bs4 import BeautifulSoup

RST_PARSE_DIRECTORY = "rst_parses/" 
ORIGINAL_TEXT_DIRECTORY = "../data/prepared_texts/" 

def write_page(forum, image_map):
  sections = "\n".join([str(v) for k,v in image_map.items() if 'section' in k])
  reviews = [str(image_map[k])
      for k in sorted(image_map.keys()) if 'review' in k]
  rebuttals = [str(image_map[k])
      for k in sorted(image_map.keys()) if 'rebuttal' in k]
  assert len(reviews) == len(rebuttals)
  for rev, reb in zip(reviews, rebuttals):
    sections += f"\n{rev}\n{reb}"

  with open("page_template.html", 'r') as f:
    html_template = f.read()

  with open(f'{forum}.html', 'w') as f:
    f.write(html_template.replace("SECTIONS", sections))
    


def main():

  images = collections.defaultdict(dict)

  for path in glob.glob(f'{RST_PARSE_DIRECTORY}/*'):
    forum = path.split("/")[-1][:-6]
    with open(path, 'r') as f:
      lines = f.readlines()

    filenames = list(sorted(glob.glob(f'{ORIGINAL_TEXT_DIRECTORY}/{forum}/*.txt')))
    assert len(lines) == len(filenames)

    titles = [filename.split("/")[-1][:-4] for filename in filenames]

    for title, line in zip(titles, lines):
      with open("temp.json", 'w') as f:
        f.write(line)
      with open('temp.html', 'w') as f:
        f.write("")
      subprocess.run(f"visualize_rst_tree temp.json temp.html --embed_d3js",
          shell=True)
      with open("temp.html", 'r') as f:
        parsed_html = BeautifulSoup(f.read(), features="html.parser")
        if parsed_html.body is None:
          continue
        images[forum][title] = parsed_html.body.find_all('script')[0]
    break
    

  for forum, image_map in images.items():
    write_page(forum, image_map)





if __name__ == "__main__":
  main()

import argparse
import subprocess

parser = argparse.ArgumentParser(
    description="Run Grobid on a particular conference's pdfs")
parser.add_argument("-p", "--path", type=str, help="Path to data directory")
parser.add_argument("-g",
                    "--grobid_path",
                    type=str,
                    help="Path to grobid directory")

def main():

  args = parser.parse_args()

  subprocess.run([
      "java",
      "-Xmx4G",
      "-jar",
      f"{args.grobid_path}/grobid-core/build/libs/grobid-core-0.7.1-onejar.jar",
      "-r",
      "-gH",
      f"{args.grobid_path}/grobid-home",
      "-dIn",
      f"{args.path}/pdfs/",
      "-dOut",
      f"{args.path}/xmls/",
      "-exe",
      "processFullText",
  ])


if __name__ == "__main__":
  main()

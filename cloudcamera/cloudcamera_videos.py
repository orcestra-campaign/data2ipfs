#!/usr/bin/env python3
import argparse
import datetime
import glob
import subprocess
from os.path import basename, normpath


def images2mpeg(glob_pattern, outfile, framerate=60, resolution="1280x960"):
    p = subprocess.run(
        [
            "/home/m/m300575/.conda/envs/main/bin/ffmpeg",
            "-framerate", str(framerate),
            "-pattern_type", "glob",
            "-i", glob_pattern,
            "-s:v", resolution,
            "-c:v", "libx264",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-y",
            outfile,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def filename2timestamp(fpath, fmt="m%y%m%d%H%M%S%f.jpg"):
    return datetime.datetime.strptime(basename(fpath), fmt)


def extract_timestamps(glob_pattern, outfile=None):
    with open(outfile, "w") as fp:
        fp.write("#Time")
        for img in sorted(glob.iglob(glob_pattern)):
            fp.write("\n" + filename2timestamp(img).isoformat())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")
    args = parser.parse_args()

    day = basename(normpath(args.inputdir))

    extract_timestamps(f"{args.inputdir}/*.jpg", f"{day}_timestamps.txt")
    images2mpeg(f"{args.inputdir}/*.jpg", f"{day}.mp4")


if __name__ == "__main__":
    main()

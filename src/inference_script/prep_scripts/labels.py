# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# Usage: python labels.py --jobs 64 --tsv <path to train.tsv>train.tsv --output-dir <destination dir> --output-name test --txt-dir  # noqa: E114,E116,W291,E501
# pylint: disable=C0114,W1514,W0611,E401

import argparse
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def get_text(line,root):  # noqa: E231
    print(line)
    print(root)
    file_name = line.split("\t")[0]
    name = file_name.split("-",2)  # noqa: E231

    print(name)
    txt_path = name[0]+"-"+name[1]+".trans.txt"
    #txt_path = line.split("\t")[0].split("-")[0:-1].replace(".flac",".trans.txt").strip() ## implies that the text filename and wav filename should be same  # noqa: E501

    txt_path = os.path.join( root , txt_path )  # noqa: E201,E202,E203
    #txt_path = filepath[0]
    print(txt_path)
    text = ''
    with open(txt_path , mode = "r", encoding="utf-8") as file_local:  # noqa: E203,E251
        text = file_local.readline().strip()

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type = str, help = "TSV file for which labels need to be generated")  # noqa: E251
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--txt-dir")
    parser.add_argument("--jobs", default=-1, type=int, help="Number of jobs to run in parallel")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tsv_file=args.tsv  # noqa: E225
    output_dir=args.output_dir  # noqa: E225
    output_name=args.output_name  # noqa: E225

    with open(tsv_file) as tsv, open(
            os.path.join(output_dir, output_name + ".ltr"), "w",encoding="utf-8"  # noqa: E231
        ) as ltr_out, open(  # noqa: E121
            os.path.join(output_dir, output_name + ".wrd"), "w",encoding="utf-8"  # noqa: E231
        ) as wrd_out:  # noqa: E125

        root = next(tsv).strip()

        if not args.txt_dir:
            args.txt_dir = root
        
        local_arr = []

        local_arr.extend(Parallel(n_jobs = args.jobs)( delayed(get_text)(line , args.txt_dir) for line in tqdm(tsv)))  # noqa: E251,E201,E203,W291 
    
        
        formatted_text = []  # noqa: E303
        for text in local_arr:
            local_list = list( text.replace(" ", "|") )  # noqa: E201,E202
            final_text = " ".join(local_list) + ' |'
            formatted_text.append(final_text)


        wrd_out.writelines("\n".join(local_arr))  # noqa: E303
        ltr_out.writelines("\n".join(formatted_text))


if __name__ == "__main__":
    main()

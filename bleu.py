import argparse
import random
from subprocess import Popen, PIPE
import os
import sys
from itertools import islice

parser = argparse.ArgumentParser(description='Script to compute BLEU')
parser.add_argument(
    "--ref", type=str, required=True,
    help="path to file with references"
)
parser.add_argument(
    "--hyp", type=str, required=True,
    help="path to file with hypotheses"
)
parser.add_argument(
    "--n", type=int, required=True,
    help="--n argument used to generate --ref file using eval.py"
)
parser.add_argument(
    "--with_contexts", dest="with_contexts", action="store_true",
    help="whether to consider contexts or not when compute BLEU"
)
parser.add_argument(
    "--bleu_path", type=str, required=True,
    help="path to mosesdecoder sentence-bleu binary"
)
parser.add_argument(
    "--mode", type=str, required=True,
    help="whether to average or take random example per word"
)
args = parser.parse_args()
assert args.mode in ["average", "random"], "--mode must be averange or random"


def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]


def read_def_file(file, n, with_contexts=False):
    defs = {}
    while True:
        lines = next_n_lines(file, n + 2)
        if len(lines) == 0:
            break
        assert len(lines) == n + 2, "Something bad in hyps file"
        word = lines[0].split("Word:")[1].strip()
        context = lines[1].split("Context:")[1].strip()
        dict_key = word + " " + context if with_contexts else word
        if dict_key not in defs:
            defs[dict_key] = []
        for i in range(2, n + 2):
            defs[dict_key].append(lines[i].strip())
    return defs


def read_ref_file(file, with_contexts=False):
    defs = {}
    while True:
        lines = next_n_lines(file, 3)
        if len(lines) == 0:
            break
        assert len(lines) == 3, "Something bad in refs file"
        word = lines[0].split("Word:")[1].strip()
        context = lines[1].split("Context:")[1].strip()
        definition = lines[2].split("Definition:")[1].strip()
        dict_key = word + " " + context if with_contexts else word
        if dict_key not in defs:
            defs[dict_key] = []
        defs[dict_key].append(definition)
    return defs


def get_bleu_score(bleu_path, all_ref_paths, d, hyp_path):
    with open(hyp_path, 'w') as ofp:
        ofp.write(d)
    read_cmd = ['cat', hyp_path]
    bleu_cmd = [bleu_path] + all_ref_paths
    rp = Popen(read_cmd, stdout=PIPE)
    bp = Popen(bleu_cmd, stdin=rp.stdout, stdout=PIPE, stderr=devnull)
    out, err = bp.communicate()
    if err is None:
        return float(out.strip())
    else:
        return None

with open(args.ref) as ifp:
    refs = read_ref_file(ifp, args.with_contexts)
with open(args.hyp) as ifp:
    hyps = read_def_file(ifp, args.n, args.with_contexts)

assert len(refs) == len(hyps), "Number of words being defined mismatched!"
tmp_dir = "/tmp"
suffix = str(random.random())
words = refs.keys()
hyp_path = os.path.join(tmp_dir, 'hyp' + suffix)
to_be_deleted = set()
to_be_deleted.add(hyp_path)

# Computing BLEU
devnull = open(os.devnull, 'w')
score = 0
count = 0
total_refs = 0
total_hyps = 0
for word in words:
    if word not in refs or word not in hyps:
        continue
    wrefs = refs[word]
    whyps = hyps[word]
    # write out references
    all_ref_paths = []
    for i, d in enumerate(wrefs):
        ref_path = os.path.join(tmp_dir, 'ref' + suffix + str(i))
        with open(ref_path, 'w') as ofp:
            ofp.write(d)
            all_ref_paths.append(ref_path)
            to_be_deleted.add(ref_path)
    total_refs += len(all_ref_paths)
    # score for each output
    micro_score = 0
    micro_count = 0
    if args.mode == "average":
        for d in whyps:
            rhscore = get_bleu_score(
                args.bleu_path, all_ref_paths, d, hyp_path)
            if rhscore is not None:
                micro_score += rhscore
                micro_count += 1
    elif args.mode == "random":
        d = random.choice(whyps)
        rhscore = get_bleu_score(args.bleu_path, all_ref_paths, d, hyp_path)
        if rhscore is not None:
            micro_score += rhscore
            micro_count += 1
    total_hyps += micro_count
    score += micro_score / micro_count
    count += 1
devnull.close()

# delete tmp files
for f in to_be_deleted:
    os.remove(f)
print("BLEU: ", score / count)
print("NUM HYPS USED: ", total_hyps)
print("NUM REFS USED: ", total_refs)

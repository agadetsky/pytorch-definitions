from source.datasets import Vocabulary
import argparse
import json

parser = argparse.ArgumentParser(description='Prepare vocabularies for model')
parser.add_argument(
    '--defs', type=str, required=True, nargs="+",
    help='location of json file with definitions.'
)
parser.add_argument(
    "--lm", type=str, required=False, nargs="+",
    help="location of txt file with text for LM pre-training"
)
parser.add_argument(
    '--same', dest='same', action='store_true',
    help="use same vocab for definitions and contexts"
)
parser.set_defaults(same=False)
parser.add_argument(
    "--save", type=str, required=True,
    help="where to save prepaired vocabulary (for words from definitions)"
)
parser.add_argument(
    "--save_context", type=str, required=False,
    help="where to save vocabulary (for words from contexts)"
)
parser.add_argument(
    "--save_chars", type=str, required=True,
    help="where to save char vocabulary (for chars from all words)"
)
args = parser.parse_args()
if not args.same and args.save_context is None:
    parser.error("--save_context required if --same didn't used")


voc = Vocabulary()
char_voc = Vocabulary()
if not args.same:
    context_voc = Vocabulary()

definitions = []
for f in args.defs:
    with open(f, "r") as infile:
        definitions.extend(json.load(infile))

if args.lm is not None:
    lm_texts = ""
    for f in args.lm:
        lm_texts = lm_texts + open(f).read().lower() + " "
    lm_texts = lm_texts.split()

    for word in lm_texts:
        voc.add_token(word)

for elem in definitions:
    voc.add_token(elem[0][0])
    char_voc.tok_maxlen = max(len(elem[0][0]), char_voc.tok_maxlen)
    for c in elem[0][0]:
        char_voc.add_token(c)
    for i in range(len(elem[1])):
        voc.add_token(elem[1][i])
    if args.same:
        for i in range(len(elem[2])):
            voc.add_token(elem[2][i])
    else:
        context_voc.add_token(elem[0][0])
        for i in range(len(elem[2])):
            context_voc.add_token(elem[2][i])


voc.save(args.save)
char_voc.save(args.save_chars)
if not args.same:
    context_voc.save(args.save_context)

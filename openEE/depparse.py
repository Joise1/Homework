#!/usr/bin/env python

import pdb
import os
import argparse
import json
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from annotation import read_dataset, write_dataset, Dependency


DEP_ROOT = 'ROOT'


def depparse(dataset, corenlp_dir):
    """
    Run dependency parsing on sentences.
    """
    nlp = StanfordCoreNLP(corenlp_dir)
    props = {'annotators': 'tokenize,pos,lemma,parse','pipelineLanguage':'en',
             'tokenize.whitespace': 'true', 'ssplit.isOneSentence': 'true',
             'outputFormat':'json'}

    for doc in tqdm(dataset.docs):
        for sent in doc.sents:
            try:
                corenlp_out = json.loads(nlp.annotate(sent.text, properties=props))
            except:
                continue  # skip errors

            assert len(corenlp_out['sentences']) == 1
            sent_corenlp = corenlp_out['sentences'][0]
            assert len(sent.tokens) == len(sent_corenlp['tokens'])
            assert [t.text for t in sent.tokens] == \
                [t['originalText'] for t in sent_corenlp['tokens']]

            for i, token in enumerate(sent.tokens):
                token_corenlp = sent_corenlp['tokens'][i]
                token.pos = token_corenlp['pos']
                token.lemma = token_corenlp['lemma']

            for dep_corenlp in sent_corenlp['basicDependencies']:
                dep = Dependency()
                dep.rel = dep_corenlp['dep']
                dep_tail = sent.tokens[dep_corenlp['dependent']-1]
                assert dep_tail.text == dep_corenlp['dependentGloss']
                dep.tail = dep_tail.token_id
                dep_tail.head_deps.append(dep)

                if dep.rel == DEP_ROOT:
                    continue

                dep_head = sent.tokens[dep_corenlp['governor']-1]
                assert dep_head.text == dep_corenlp['governorGloss']
                dep.head = dep_head.token_id
                dep_head.tail_deps.append(dep)

    nlp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, metavar='str',
                        default='out/semcor.json.gz',
                        help="input file (gzipped JSON)")
    parser.add_argument('--corenlp_dir', type=str, metavar='str',
                        default='./stanford-corenlp-full-2018-10-05',
                        help="Stanford CoreNLP directory")
    parser.add_argument('--output', type=str, metavar='str',
                        default='out/semcor_corenlp.json.gz',
                        help="output file (gzipped JSON)")
    args = parser.parse_args()

    if not os.path.exists('out'):
        os.mkdir('out')

    print(f"Loading the dataset from {args.input} ...")
    dataset = read_dataset(args.input)

    print("Parsing the dataset ...")
    depparse(dataset, args.corenlp_dir)

    print("Parsing is done.")

    write_dataset(dataset, args.output)
    print(f"Saved output to {args.output}")

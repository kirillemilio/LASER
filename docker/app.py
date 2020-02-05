#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
import json
import subprocess
import fastBPE
from collections import namedtuple
import logging
import asyncio
import socket
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Set, Union, Any, Optional
import numpy as np
from LASER.source.lib.text_processing import Token, BPEfastApply
from LASER.source.embed import *


assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

FASTBPE = LASER + '/tools-external/fastBPE/fast'
MOSES_BDIR = LASER + '/tools-external/moses-tokenizer/tokenizer/'
MOSES_TOKENIZER = MOSES_BDIR + 'tokenizer.perl -q -no-escape -threads 20 -l '
MOSES_LC = MOSES_BDIR + 'lowercase.perl'
NORM_PUNC = MOSES_BDIR + 'normalize-punctuation.perl -l '
DESCAPE = MOSES_BDIR + 'deescape-special-chars.perl'
REM_NON_PRINT_CHAR = MOSES_BDIR + 'remove-non-printing-char.perl'

# Romanization (Greek only)
ROMAN_LC = 'python3 ' + LASER + '/source/lib/romanize_lc.py -l '

# Mecab tokenizer for Japanese
MECAB = LASER + '/tools-external/mecab'

Model = namedtuple('Model', ['tokenizer', 'encoder', 'bpe'])


LOGGER = logging.getLogger(__name__)
MODEL = None
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


def tokenize(data,
             lang='en',
             lower_case=True,
             romanize=False,
             descape=False,
             verbose=False, over_write=False, gzip=False):
    assert lower_case, 'lower case is needed by all the models'
    assert not over_write, 'over-write is not yet implemented'

    roman = lang if romanize else 'none'
    # handle some iso3 langauge codes
    if lang in ('cmn', 'wuu', 'yue'):
        lang = 'zh'
    if lang in ('jpn'):
        lang = 'ja'
    if verbose:
        LOGGER.info(f" - Tokenizer: in language {lang} "
                    + f"{'(gzip)' if gzip else ''}"
                    + f" {'(de-escaped)' if descape else ''}"
                    + f" {'(romanized)' if romanize else ''}")

    p = subprocess.Popen([
        REM_NON_PRINT_CHAR
        + '|' + NORM_PUNC + lang
        + ('|' + DESCAPE if descape else '')
        + '|' + MOSES_TOKENIZER + lang
        + ('| python3 -m jieba -d ' if lang == 'zh' else '')
        + ('|' + MECAB + '/bin/mecab -O wakati -b 50000 ' if lang == 'ja' else '')
        + '|' + ROMAN_LC + roman],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=dict(os.environ, LD_LIBRARY_PATH=MECAB + '/lib')
    )
    p.stdin.write(data.encode())
    p.stdin.close()
    return p.stdout.read().decode()


def init_sentence_encoder():
    global MODEL
    model_dir = Path(__file__).parent / "LASER" / "models"
    encoder_path = model_dir / "bilstm.93langs.2018-12-26.pt"
    bpe_codes = str(model_dir / "93langs.fcodes")
    LOGGER.info(f' - Encoder: loading {encoder_path}')

    encoder = SentenceEncoder(encoder_path,
                              max_sentences=None,
                              max_tokens=12000,
                              sort_kind='mergesort',
                              cpu=True)
    print(encoder)
    bpe = None
    print(bpe_codes.replace('fcodes', 'fvocab'))
    bpe = fastBPE.fastBPE(bpe_codes, bpe_codes.replace('fcodes',
                                                       'fvocab'))
    print(bpe)
    MODEL = Model(bpe=bpe, encoder=encoder, tokenizer=tokenize)


@app.route("/vectorize")
def vectorize():
    data = request.args.get('q')
    lang = request.args.get('lang')
    if lang is None or not lang:
        lang = "en"
    output = MODEL.tokenizer(
        data=data,
        lang=lang,
        romanize=True if lang == 'el' else False,
        lower_case=True,
        gzip=False,
        verbose=True,
        over_write=False)
    output = MODEL.bpe.apply([output])
    output = MODEL.encoder.encode_sentences(output).flatten()
    body = {'content': data, 'embedding': output.tolist()}
    return jsonify(body)


if __name__ == "__main__":
    init_sentence_encoder()
    app.run(debug=True, port=80, host='0.0.0.0')

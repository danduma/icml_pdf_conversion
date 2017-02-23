# Convert a PDF or set of PDFs to files for CTR
#
# Copyright:   (c) Daniel Duma 2016
# Author: Daniel Duma <danielduma@gmail.com>

# For license information, see LICENSE.TXT

# Usage:
#  python pdfconversion.py -p icml2016/*.pdf -t icml2016/txt_files -m pdf2txt
#  python pdfconversion.py -t icml2016/txt_files -m txt2dat

import os
from os import path
import glob
import argparse
import re
import codecs

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from cStringIO import StringIO

MAX_VOCAB = 20000
MIN_DF=0.05
MAX_DF=0.5

def pdf_to_text(pdfname):
    # PDFMiner boilerplate
    # from https://gist.github.com/jmcarp/7105045
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = file(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text

def ensureDirExists(dir):
    # type: (basestring) -> None
    """
        Makes sure directory exists. If not, it creates it.
    """
    dir=path.normpath(dir)
    if not path.isdir(dir):
        try:
            os.makedirs(dir)
        except:
            print("Failed to create directory %s" % dir)


def pdfToTxt(file_path, output_dir):
    """
        Converts a single pdf file to txt, stores it with the same file name but
        .txt extension in the output_dir

        :param file_path: path to file on file system
        :param output_dir: path to directory. If not existing, it will create it.
    """
    file_path=path.abspath(file_path)
    file_name, file_ext=path.splitext(path.basename(file_path))
    output_file_name=file_name+".txt"
    output_file_path=path.join(output_dir,output_file_name)

##    with open(file_path) as f:
##        doc=slate.PDF(f)

    print "Converting %s -> %s" % (file_path, output_file_path)

    try:
        text=pdf_to_text(file_path)
    except:
        print "Error converting",file_path
        return

    with open(output_file_path,"w") as f:
        f.write(text)


def convertFiles(input_mask, output_dir):
    """
        Converts all files matching the mask
    """
    input_dir=path.abspath(input_mask)
    for file_path in glob.glob(input_mask):
        pdfToTxt(file_path, output_dir)

def loadVocab(vocab_path):
    """
        Loads an existing vocabulary
    """
##    vocab={}
##    with open(vocab_path) as f:
##        for index,line in enumerate(f):
##            term=line.strip().strip("\n")
##            vocab[term]=index
##    return vocab

    f=codecs.open(path.join(file_dir,"vocab.dat"),encoding="utf-8",errors="ignore",mode="r")
    text=" ".join(f.readlines())
    text=re.sub(r"\r?\n"," ",text)
    return text

def isNumber(text):
    """
    """
    try:
        num=float(text)
    except:
        return False
    return True

def cleanVocabulary(vocab, matrix):
    """
        Removes features from the matrix based on criteria such as length
    """

    mask=[]
    for index,feature in enumerate(vocab):
        if isNumber(feature) or len(feature) > 20:
            continue
        mask.append(index)

##    MIN_VALUE=0
##    if len(vocab) > MAX_VOCAB:
##        freqs=np.asarray(matrix.sum(axis=0)).ravel()
##        top=np.sort(freqs)
##        top.sort()
##        top[:]=top[::-1]
##        MIN_VALUE=top[MAX_VOCAB-1]
##
##    for index,feature in enumerate(vocab):
##        if mask[index] and matrix[:,index].sum(axis=0) < MIN_VALUE:
##            mask[index] = False
    keep=[(index in mask) for index in range(len(vocab))]

    for index,keep_item in enumerate(keep):
        if not keep_item:
            vocab.pop(index)

    return matrix[:,mask], vocab

def createMatrixForCTR(file_dir, vocab_path, ignore=None):
    """
        Creates the input files in the format that CTR expects
    """
    if vocab_path is not None:
        vocab_file=loadVocab(vocab_path)
    else:
        vocab_file=""
    files_text=[vocab_file]
    files_names=["vocab"]

    input_mask=path.join(file_dir,"*.txt")
    for file_path in glob.glob(input_mask):
        if ignore:
            if re.search(ignore,file_path,re.IGNORECASE):
                continue

        with codecs.open(path.abspath(file_path),encoding="utf-8",errors="ignore") as f:
            text=" ".join(f.readlines())
            text=re.sub(r"\r?\n"," ",text) # remove line endings
            files_text.append(text)
            files_names.append(file_path)

    vectorizer=CountVectorizer('content',
                               decode_error="ignore",
                               strip_accents="unicode",
                               stop_words="english",
                               max_features=MAX_VOCAB,
                               min_df=MIN_DF, # minimum portion of documents feature should appear in
                               max_df=MAX_DF
                               )

    data=vectorizer.fit_transform(files_text).toarray()
    vocab=vectorizer.get_feature_names()
    # TODO cleanVocabulary doesn't seem to be working the way I'd want it to. Some bug somewhere
##    data, vocab=cleanVocabulary(vectorizer.get_feature_names(),data)
    out_mult=codecs.open(path.join(file_dir,"mult.dat"),mode="w",encoding="utf-8",errors="ignore")

    for index,row in enumerate(data):
        total_terms=0
        line=str(index)
        for index2,val in enumerate(row):
            if val > 0:
                line+=" %d:%d" % (index2,val)
                total_terms+=1
        line=unicode(total_terms)+" "+line
        out_mult.write(line+"\n")
    out_mult.close()

    out_vocab=codecs.open(path.join(file_dir,"vocab.dat"),mode="w",encoding="utf-8",errors="ignore")
    for index,feature in enumerate(vocab):
        out_vocab.write(feature+"\n")
    out_vocab.close()

    out_files=codecs.open(path.join(file_dir,"files.dat"),mode="w",encoding="utf-8",errors="ignore")
    for index,fn in enumerate(files_names):
        out_files.write(fn+"\n")
    out_files.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdfdir', '-p', type=str, default='*.pdf',
                        help='Input file or directory followed by wildcard')
    parser.add_argument('--txtdir', '-t', type=str,
                        default='.',
                        help='Directory for .txt and .dat files')
    parser.add_argument('--vocab', '-v', type=str,
                        default=None,
                        help='Path to existing vocab file to use as input')
    parser.add_argument('--mode', '-m', type=str, choices=["full","pdf2txt","txt2dat"], default=["full"],
                        help='Input file or directory followed by wildcard')
    parser.add_argument('--ignore', '-i', type=str, default="-supp",
                        help='Regex: Files matching pattern will be ignored')
    args = parser.parse_args()

    ensureDirExists(path.abspath(args.txtdir))

    if args.mode in ["full","pdf2txt"]:
        convertFiles(args.pdfdir,args.txtdir)

    if args.mode in ["full","txt2dat"]:
        createMatrixForCTR(args.txtdir, args.vocab)


if __name__ == '__main__':
    main()


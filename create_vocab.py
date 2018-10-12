#!/usr/bin/python
import re
import sys
from collections import Counter
import zipimport
importer = zipimport.zipimporter('nltk.zip')
nltk = importer.load_module('nltk')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
# DATA_FILE_NAME = "data/DBPedia.verysmall.hdfs/verysmall_train.txt"
DATA_FILE_NAME = "data/DBPedia.full.hdfs/full_train.txt"
STOPWORDS_FILE_NAME = "stopwords.txt"
VOCAB_FILE_NAME = "vocab.txt"

def remove_till_first_quote(text):
    regex = r"^(.*?)\""
    text = re.sub(regex, '', text)
    return text

def remove_unicode(text):
    """Replace unicode codes like \uxxxx with space"""
    regex = r"(\\u....)"
    text = re.sub(regex, ' ', text)
    return text

def denoise_text(text):
    text = remove_till_first_quote(text)
    text = remove_unicode(text)
    text = remove_punctuation(text)
    return text

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(text):
    """Remove punctuation and replace with space"""
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

stopwords = []
f = open(STOPWORDS_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    if(line):
        stopwords.append(line)
f.close()


dataFile = open(DATA_FILE_NAME, "r")

allWords = set()
for line in dataFile.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    document = splitLine[1]
    document = denoise_text(document)
    words = document.split()
    words = to_lowercase(words)
    for word in words:
        if word in stopwords:
            continue
        allWords.add(word)

vocabFile = open(VOCAB_FILE_NAME, "w")
allWordsList = list(allWords)
allWordsList.sort()
wordId = 0
for word in allWordsList:
    vocabFile.write(word + '\t' + str(wordId) + '\n')
    wordId += 1
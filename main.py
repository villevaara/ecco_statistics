from libcommon.octavo_api_client import (
    OctavoEccoClient
    )
from libcommon.metadata_readers import (
    load_good_metadata
    )
# from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
import random
# import sys
import csv
import re
from collections import OrderedDict


def get_doctext_tokens(doctext):
    text_tokens = word_tokenize(doctext)
    return text_tokens


def get_stemmed_tokens(doctext_tokens, stemmer):
    stemmed_tokens = []
    for token in doctext_tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens


def get_stemmed_token(token, stemmer):
    stemmed_token = (stemmer.stem(token))
    return stemmed_token


def get_stemmed_baseline(wordfile):
    with open(wordfile) as word_file:
        english_words = set(word.strip().lower() for word in word_file)
    stemmed_baseline = set()
    for word in english_words:
        stemmed_baseline.add(stemmer.stem(word))
    return stemmed_baseline


def get_stemmed_baseline2(wordfile):
    with open(wordfile) as word_file:
        english_words = (
            set(word.split("\t")[0].strip().lower() for word in word_file))
    stemmed_baseline = set()
    for word in english_words:
        stemmed_baseline.add(stemmer.stem(word))
    return stemmed_baseline


def evaluate_text_ocr(doctext, baseline_set, stemmer, min_length=3):
    retdict = {'recognised': [], 'unrecognised': []}
    tokens = get_doctext_tokens(doctext)
    for i in range(0, len(tokens)):
        token = tokens[i]
        if len(token) < min_length:
            continue
        if re.search('[a-zA-Z]', token) is None:
            continue
        stemmed_token = get_stemmed_token(token, stemmer)
        if stemmed_token in baseline_set:
            retdict['recognised'].append(token)
        else:
            retdict['unrecognised'].append(token)
    return retdict


def get_metadata_yearsubset(good_metadata, year):
    results_metadata = {}
    for key, value in good_metadata.items():
        if value.get('estc_publication_year') == str(year):
            results_metadata[key] = value
    return results_metadata


def get_metadata_langsubset(good_metadata, lang):
    results_metadata = {}
    for key, value in good_metadata.items():
        if value.get('estc_language') == lang:
            results_metadata[key] = value
    return results_metadata


def get_metadata_sample(metadata, sample_size):
    metadata_list = list(metadata.values())
    if len(metadata_list) < sample_size:
        sample_size = len(metadata)
    random_sample = random.sample(metadata_list, sample_size)
    return random_sample


def write_wordset(words, outputfile):
    output = set(words)
    outdict = {}
    for word in output:
        outdict[word] = words.count(word)
    sorteddict = OrderedDict(
        sorted(outdict.items(), key=lambda t: t[1], reverse=True))
    with open(outputfile, 'w') as outfile:
        csvwriter = csv.writer(outfile)
        for key, value in sorteddict.items():
            csvwriter.writerow([key, value])
    # with open(outputfile, 'w') as outfile:
    #     for word in output:
    #         outfile.write(word + "\n")


def get_lensums(wordlist):
    words = wordlist
    lens = []
    for word in words:
        lens.append(len(word))
    max_len = max(lens)
    lensums = {}
    for word_length in range(2, max_len + 1):
        lensums[word_length] = lens.count(word_length)
    return lensums


def get_yearly_results(good_metadata, year, sample_size, eccoclient):
    year_metadata_subset = get_metadata_yearsubset(
        good_metadata, year)
    year_metadata_subset = get_metadata_langsubset(
        year_metadata_subset, "English")
    year_metadata_sample = get_metadata_sample(
        year_metadata_subset, sample_size)

    yearly_sample_eccoids = []
    for item in year_metadata_sample:
        yearly_sample_eccoids.append(item.get('ecco_id'))

    print("Fetching n eccoids: " + str(len(yearly_sample_eccoids)))

    yearly_results_by_doc = []
    for docid in yearly_sample_eccoids:
        doctext = eccoclient.get_text_for_document_id(docid).get('text')
        text_evaluated = evaluate_text_ocr(doctext, stemmed_baseline, stemmer)
        good_words = text_evaluated.get('recognised')
        bad_words = text_evaluated.get('unrecognised')
        ratio = len(good_words) / (len(good_words) + len(bad_words))
        yearly_results_by_doc.append({'rec_words': good_words,
                                      'unk_words': bad_words,
                                      'good_ratio': ratio})

    rec_words = []
    unk_words = []
    print("Combining yearly results.")
    for yearly_result in yearly_results_by_doc:
        rec_words.extend(yearly_result.get('rec_words'))
        unk_words.extend(yearly_result.get('unk_words'))
    ratio = len(rec_words) / (len(rec_words) + len(unk_words))

    rec_words_lensum = get_lensums(rec_words)
    unk_words_lensum = get_lensums(unk_words)

    yearly_results_all = {'rec_words': rec_words,
                          'unk_words': unk_words,
                          'ratio': ratio,
                          'rec_sums': rec_words_lensum,
                          'unk_sums': unk_words_lensum}
    return yearly_results_all


def write_yeartable_header(yeartable_csvfile):
    with open(yeartable_csvfile, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_row = ['year', 'ratio', 'rec', 'unk', ]
        rec_len_list = []
        unk_len_list = []
        for i in range(2, 51):
            rec_len_list.append("rec-" + str(i))
            unk_len_list.append("unk-" + str(i))
        header_row.extend(rec_len_list)
        header_row.extend(unk_len_list)
        csvwriter.writerow(header_row)


# script start

eccoclient = OctavoEccoClient()
# stemmer = PorterStemmer()
# stemmer = LancasterStemmer()
stemmer = SnowballStemmer('english')
# wordfile = "data/wordsEn.txt"
# wordfile = "data/words.txt"
wordfile = "data/count_1w.txt"
# http://norvig.com/ngrams/
stemmed_baseline = get_stemmed_baseline2(wordfile)
good_metadata = load_good_metadata(
    "../data-own-common/good_metadata.json")
yeartable_csvfile = "output/allyears_newest.csv"
start_new = True
yearly_sample_size = 1000000
years_to_evaluate = (list(range(1700, 1799)))

if start_new:
    write_yeartable_header(yeartable_csvfile)

for year in years_to_evaluate:
    print("processing year: " + str(year))
    yearly_results = get_yearly_results(
        good_metadata, year, yearly_sample_size, eccoclient)

    with open(yeartable_csvfile, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        outrow = [year,
                  yearly_results.get('ratio'),
                  len(yearly_results.get('rec_words')),
                  len(yearly_results.get('unk_words')),
                  ]
        rec_len_list = []
        unk_len_list = []
        for i in range(2, 51):
            rec_len = yearly_results.get('rec_sums').get(i)
            if rec_len is None:
                rec_len = 0
            rec_len_list.append(rec_len)
            unk_len = yearly_results.get('unk_sums').get(i)
            if unk_len is None:
                unk_len = 0
            unk_len_list.append(unk_len)
        outrow.extend(rec_len_list)
        outrow.extend(unk_len_list)
        csvwriter.writerow(outrow)

    print("finished with year: " + str(year))
    # print("Writing word totals for year " + str(year))
    outfile = "output/" + str(year)
    # write_wordset(yearly_results['rec_words'], (outfile + "-rec.txt"))
    # write_wordset(yearly_results['unk_words'], (outfile + "-unk.txt"))

# rec_words_outfile = "output/recset.txt"
# write_wordset(yearly_results.get('rec_words'), rec_words_outfile)
# unk_words_outfile = "output/unkset.txt"
# write_wordset(yearly_results.get('unk_words'), unk_words_outfile)

# docid = "0145100106"
# doctext = eccoclient.get_text_for_document_id(docid).get('text')

# print("good: " + str(len(good_words)))
# print("bad:  " + str(len(bad_words)))

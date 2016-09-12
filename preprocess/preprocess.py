import nltk
import csv
import string
from tqdm import *

stop_words = "StopWords_Eng-Ind.txt"
not_spam_csv = "../data/NOT_SPAM.csv"
spam_csv = "../data/SPAM.csv"
preprocessed_data_csv = "../data/PREPROCESSED_DATA.csv"
preprocessed_data_txt = "../data/PREPROCESSED_DATA.txt"
preprocessed_data_spam_csv = "../data/PREPROCESSED_DATA_SPAM.csv"
preprocessed_data_spam_txt = "../data/PREPROCESSED_DATA_SPAM.txt"
preprocessed_data_not_spam_csv = "../data/PREPROCESSED_DATA_NOT_SPAM.csv"
preprocessed_data_not_spam_txt = "../data/PREPROCESSED_DATA_NOT_SPAM.txt"

# Mendapatkan set dari kata-kata stopword yang didefinisikan di file teks
def get_stopwords():
    sw = open(stop_words,encoding = 'utf-8', mode = 'r'); stop = sw.readlines(); sw.close()
    stop = [word.strip() for word in stop]; stop = set(stop)
    return stop

# Mendapatkan set dari simbol simbol tanda baca yang akan dibuang (diabaikan)
def get_punctuations():
    punctuations = set(string.punctuation).union(["''","...","``",".."])
    return punctuations

punctuations = get_punctuations()
stop = get_stopwords()

# Menggabungkan kata-kata yang akan dibuang dari dataset
removed_words = stop.union(punctuations)

# Membuka csv file not spam, disimpan dalam list
with open(not_spam_csv, newline = '') as csvfile:
    reader = csv.reader(csvfile, quotechar = '"')
    not_spam_list = list(reader)

# Membuka csv file spam, disimpan dalam list
with open(spam_csv, newline = '') as csvfile:
    reader = csv.reader(csvfile, quotechar='"')
    spam_list = list(reader)

preprocessed_data_spam = []
preprocessed_data_not_spam = []

print("Processing. . .")
# Tokenize list not spam dan spam, lalu membuang kata-kata yang telah didefinisikan akan dibuang
# Dimasukkan ke list data yang akan menjadi output
print("Melakukan pra-proses not spam")
for row in tqdm(not_spam_list):
    sms = nltk.word_tokenize(''.join(row).lower())
    sms = [word for word in sms if word not in removed_words]
    preprocessed_data_not_spam.append(' '.join(sms))

print("Melakukan pra-proses spam")
for row in tqdm(spam_list):
    sms = nltk.word_tokenize(''.join(row).lower())
    sms = [word for word in sms if word not in removed_words]
    preprocessed_data_spam.append(' '.join(sms))

# Menyimpan list data yang telah di pra-proses ke file teks
print("Menyimpan list data yang telah di pra-proses ke file teks")

output = open(preprocessed_data_not_spam_txt, "w")
for sms in tqdm(preprocessed_data_not_spam):
    output.write(sms+"\n")
output.close()

output = open(preprocessed_data_spam_txt, "w")
for sms in tqdm(preprocessed_data_spam):
    output.write(sms+"\n")
output.close()

# Menyimpan list data yang telah di pra-proses ke file csv
print("Menyimpan list data yang telah di pra-proses ke file csv")

with open(preprocessed_data_not_spam_csv, 'w', newline = '') as f:
    writer = csv.writer(f)
    for sms in tqdm(preprocessed_data_not_spam):
        writer.writerow([sms])

with open(preprocessed_data_spam_csv, 'w', newline = '') as f:
    writer = csv.writer(f)
    for sms in tqdm(preprocessed_data_spam):
        writer.writerow([sms])

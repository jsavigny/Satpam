import csv
import arff
from tqdm import *

preprocessed_data_spam_csv = "../data/PREPROCESSED_DATA_SPAM.csv"
preprocessed_data_spam_txt = "../data/PREPROCESSED_DATA_SPAM.txt"
preprocessed_data_not_spam_csv = "../data/PREPROCESSED_DATA_NOT_SPAM.csv"
preprocessed_data_not_spam_txt = "../data/PREPROCESSED_DATA_NOT_SPAM.txt"
arff_output = "../data/arff_output.arff"

# TODO: weighting word?
def weight(word):
    print(word)

# Membaca file praproses not spam
with open(preprocessed_data_not_spam_csv, newline = '') as csvfile:
    reader = csv.reader(csvfile, quotechar = '"')
    not_spam_list = list(reader)

# Membaca file praproses spam
with open(preprocessed_data_spam_csv, newline = '') as csvfile:
    reader = csv.reader(csvfile, quotechar = '"')
    spam_list = list(reader)

# Mengubah menjadi per kata
not_spam_list = [''.join(sms).split() for sms in not_spam_list]
spam_list = [''.join(sms).split() for sms in spam_list]

the_list = not_spam_list + spam_list

# Membuat himpunan bag-of-words
bags = set()

for sms in the_list:
    print(set(sms))
    bags = bags.union(set(sms))

# Membuat dataset untuk dijadikan arff berdasarkan model bag-of-words
arff_list = []

for sms in tqdm(not_spam_list):
    temp_list = [1 if x in sms else 0 for x in bags] # Belum menggunakan weight, hanya keberadaan kata
    temp_list.append('not_spam')
    arff_list.append(temp_list)

for sms in tqdm(spam_list):
    temp_list = [1 if x in sms else 0 for x in bags] # Belum menggunakan weight, hanya keberadaan kata
    temp_list.append('spam')
    arff_list.append(temp_list)

# Membuat atribut untuk arff
attributes = [('word'+str(list(bags).index(word)),'REAL') for word in bags]
attributes.append(('spam?',['spam','not_spam']))

# Membuat objek untuk dijadikan arff dengan library liac-arff
obj = {
   'description': u'Bag of Words sms spam filter',
   'relation': 'words',
   'attributes': attributes,
   'data': arff_list,
}

# Membuat file arff
output = open(arff_output, "w+")
output.write(arff.dumps(obj))
output.close()

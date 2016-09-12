import csv
import arff
from tqdm import *
import math

preprocessed_data_spam_csv = "../data/PREPROCESSED_DATA_SPAM.csv"
preprocessed_data_spam_txt = "../data/PREPROCESSED_DATA_SPAM.txt"
preprocessed_data_not_spam_csv = "../data/PREPROCESSED_DATA_NOT_SPAM.csv"
preprocessed_data_not_spam_txt = "../data/PREPROCESSED_DATA_NOT_SPAM.txt"
arff_output = "../data/arff_output.arff"

# TODO: weighting word?
def weight(word):
    print(word)

def tf(word, document):
    return document.count(word)/len(document)

def n_containing(word, documents):
    return sum(1 for document in documents if word in document)

def idf(word, documents):
    return math.log(len(documents) / (1 + n_containing(word, documents)))

def tfidf(word, document, documents):
    return tf(word, document) * idf(word, documents)

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
    #temp_list = [tfidf(word, sms, the_list) for word in bags] # Dipakai jika ingin pake weight tf idf (WARNING, LAMA)
    temp_list = [1 if word in sms else 0 for word in bags] # Dipakai jika ingin pake binary occurences
    #temp_list = [sms.count(word) for word in bags] # Dipakai jika ingin pake frekuensi kemunculan kata
    #temp_list.append(len(sms)) # Dipakai jika ingin pake banyaknya kata
    temp_list.append('not_spam')
    arff_list.append(temp_list)

for sms in tqdm(spam_list):
    #temp_list = [tfidf(word, sms, the_list) for word in bags]
    temp_list = [1 if word in sms else 0 for word in bags]
    #temp_list = [sms.count(word) for word in bags]
    #temp_list.append(len(sms))
    temp_list.append('spam')
    arff_list.append(temp_list)

# Membuat atribut untuk arff
attributes = [('word'+str(list(bags).index(word)),'REAL') for word in bags]
#attributes.append(('size','REAL')) # Dipakai jika ingin pake banyaknya kata
attributes.append(('spam?',['spam','not_spam']))

# Membuat objek untuk dijadikan arff dengan library liac-arff
obj = {
   'description': u'Bag of Words sms spam filter',
   'relation': 'words',
   'attributes': attributes,
   'data': arff_list,
}

#print(arff.dumps(obj))

# Membuat file arff
output = open(arff_output, "w")
output.write(arff.dumps(obj))
output.close()

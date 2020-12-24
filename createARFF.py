from sklearn.datasets import fetch_20newsgroups
import csv
import collections
import copy
import math

categoriesraw = ['sci.med', 'sci.space', 'talk.politics.misc', 'comp.sys.mac.hardware', 'rec.sport.baseball']
categories = ['medicine', 'space', 'politicsmisc', 'machardware', 'sportbaseball']
categoryTranslation = {}
assert len(categoriesraw) == len(categories)
for i in range(len(categoriesraw)):
    categoryTranslation[categoriesraw[i]] = categories[i]

def importDocs():
    newsgroups_train = fetch_20newsgroups(subset='all')


    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                          categories=categoriesraw)

    print(newsgroups_train.target_names)
    print("Number of texts = " + str(len(newsgroups_train.data)))
    D = newsgroups_train.filenames.shape[0]
    vectors = []

    categories = newsgroups_train.target_names

    for i in range(D):
        doc = newsgroups_train.data[i]
        cat = newsgroups_train.target[i]
        vector = ["docid=" + str(i), categories[cat]]

        words = doc.replace("\n", ' ')
        words.encode('unicode_escape')
        words = words.replace(".", "")
        words = words.replace("(", " ")
        words = words.replace(")", " ")
        words = words.replace("'s", "")
        words = words.replace("/", " ")
        words = words.replace("'", "")
        words = words.replace("==", "")
        words = words.replace("--", "")
        words = words.replace("__", "")
        words = words.replace("*", "")
        words = words.replace(">", "")
        words = words.replace("<", "")
        words = words.replace(",", " ")
        words = words.replace(";", " ")
        words = words.replace("?", "")
        words = words.replace("¿", "")
        words = words.replace("``", "")
        words = words.replace("#", "")
        words = words.replace("^", "")
        words = words.replace("|", "")
        words = words.replace("\t", '')
        words = words.replace("\r", '')
        words = words.replace("-:", "")
        words = words.replace(":", " ")
        words = words.replace("!", "")
        words = words.replace("¡", "")
        words = words.replace("\"", "")
        words = words.replace("%", "")
        words = words.replace("{", "")
        words = words.replace("}", "")
        words = words.replace("º", "")
        words = words.lower()
        words = words.split(" ")
        words = list(filter(lambda a: a != '' and a != " " and a != "-" and a != '"' and a != '\x0c', words))

        counts = [0, 0]
        total = 0
        for word in words:
            total += 1
            if word not in vector:
                vector.append(word)
                counts.append(1)
            else:
                # pass #BINOMIAL
                counts[vector.index(word)] += 1  # MULTINOMIAL
        # counts = [a/total for a in counts]
        assert len(vector) + 2 == len(counts) + 2

        vectors.append((vector, counts))

    print("Number of vectors: " + str(len(vectors)))
    return vectors

def createCollectionFrequency(vectors, categories):
    catvocab = {}
    for cat in categories:
        if cat not in catvocab: catvocab[cat] = {}

    for vector in vectors:
        cat = vector[0][1]
        words = vector[0][2:]
        counts = vector[1][2:]

        for i in range(len(words)):
            if words[i] not in catvocab[cat]:
                catvocab[cat][words[i]] = counts[i]
            else:
                catvocab[cat][words[i]] += counts[i]
    return catvocab

def createDocumentFrequency(vectors, categories):
    catvocab = {}
    for cat in categories:
        if cat not in catvocab: catvocab[cat] = {}

    for vector in vectors:
        cat = vector[0][1]
        words = vector[0][2:]
        counts = vector[1][2:]

        for i in range(len(words)):
            if words[i] not in catvocab[cat]:
                catvocab[cat][words[i]] = 1
            else:
                catvocab[cat][words[i]] += 1
    return catvocab

def createMutualInformation(vectors, categories):
    wholevocab = []
    catfeatures = {}
    for vector in vectors:
        words = vector[0][2:]
        for word in words:
            if word not in wholevocab: wholevocab.append(word)
    print("NUMBER OF WORDS: " + str(len(wholevocab)))
    print("Calculating mutual information indexes: ")
    fob = open("MIdata.txt", "w")
    writer = csv.writer(fob, delimiter=",")
    for cat in categories:
        writer.writerow(["Category", cat])
        print("Calculating features for category: " + cat)
        catfeatures[cat] = {}
        counter = 0
        for term in wholevocab:
            if counter%100 == 0: print("Done " + str(counter/len(wholevocab)*100))
            A = calculateMI(vectors, cat, term)
            catfeatures[cat][term] = A
            writer.writerow([term, A])
            counter += 1
    return catfeatures



def calculateMI(vectors, targetclass, term):
    numberOfDocumentsWithT = 0
    numberOfDocumentsWithC = 0
    numberOfDocumentsWithoutT = 0
    numberOfDocumentsWithoutC = 0
    numberOfDocumentsWithTandC = 0
    numberOfDocumentsWithTandNotC = 0
    numberOfDocumentsWithoutTandC = 0
    numberOfDocumentsWithoutTandnotC = 0

    numberOfDocuments = len(vectors)
    counter = 0
    for vector in vectors:
        counter += 1
        cat = vector[0][1]
        words = vector[0][2:]

        if term in words:
            numberOfDocumentsWithT += 1
            if cat == targetclass:
                numberOfDocumentsWithC += 1
                numberOfDocumentsWithTandC += 1
            else:
                numberOfDocumentsWithoutC += 1
                numberOfDocumentsWithTandNotC += 1
        else:
            numberOfDocumentsWithoutT += 1
            if cat == targetclass:
                numberOfDocumentsWithC += 1
                numberOfDocumentsWithoutTandC += 1
            else:
                numberOfDocumentsWithoutC += 1
                numberOfDocumentsWithoutTandnotC += 1

    conditionals = [[numberOfDocumentsWithoutTandnotC, numberOfDocumentsWithoutTandC], [numberOfDocumentsWithTandNotC, numberOfDocumentsWithTandC]]
    forT = [numberOfDocumentsWithoutT, numberOfDocumentsWithT]
    forC = [numberOfDocumentsWithC, numberOfDocumentsWithoutC]
    cumulator = 0
    for t in [0, 1]:
        for c in [0, 1]:
            try:
                cumulator += conditionals[t][c]/numberOfDocuments*math.log((conditionals[t][c]*numberOfDocuments/(forT[t]*forC[c])),2)
            except:
                cumulator += float("-inf")

    return cumulator




def chooseVocab(vocabulary, k):
    backup = copy.deepcopy(vocabulary)
    ret = []
    step = 1
    wordcounters = {}
    for cat in vocabulary:
        v = vocabulary[cat]
        vcounter = collections.Counter(v)
        wordcounters[cat] = vcounter
    addwords = 0
    while addwords < k:
        for cat in vocabulary:
            added = 0
            while added < step:
                max_word = max(vocabulary[cat], key = vocabulary[cat].get)
                if max_word in ret:
                    vocabulary[cat].pop(max_word)
                else:
                    ret.append(max_word)
                    vocabulary[cat].pop(max_word)
                    added += 1
                    addwords += 1

    assert len(ret) == k

    vocabulary = copy.deepcopy(backup)

    return ret

def getClassFeatures(vocabulary, k):
    backup = copy.deepcopy(vocabulary)
    vocab = {}
    for cat in categoriesraw:
        vocab[cat]= {}

    for cat in vocabulary:
        counter = 0
        while counter < k:
            max_word = max(vocabulary[cat], key=vocabulary[cat].get)
            vocab[cat][max_word] = vocabulary[cat][max_word]
            vocabulary[cat].pop(max_word)
            counter += 1

    return vocab

def createARFF(vectors, vocab, k):
    multinomial = True
    r = open("/Users/Ernesto/PycharmProjects/20NewsGroups/Data.arff", "w")
    csvwriter = csv.writer(r, delimiter=",", quoting = csv.QUOTE_NONE, escapechar = " ")

    csvwriter.writerow(["@relation 20NewsGroups"])

    wordcounter = 0
    for word in vocab:
        #if word == "class": word = "clas"
        wordcounter+=1
        csvwriter.writerow(["@attribute " + word + " integer"])
    print("Number of attributes created = " + str(wordcounter))

    csvwriter.writerow(
        [str("@attribute targetclass {medicine,space,politicsmisc,machardware,sportbaseball}")])
    csvwriter.writerow(["@data"])
    textcounter = 0
    for text in vectors:
        textcounter += 1
        v = []
        cat = text[0][1]
        words = text[0][2:]
        counts = text[1][2:]
        bitsChecker = 0
        for word in vocab:
            bitsChecker += 1
            if word in words:
                if multinomial: v.append(counts[words.index(word)])
                else: v.append(1)
            else: v.append(0)
        assert len(v) == k
        v.append(categoryTranslation[cat])
        csvwriter.writerow(v)
    r.close()
    print("Texts loaded = " + str(textcounter))

def encodetoascii(origin, dest):
    for line in origin:
        dest.write(line.encode("ascii", "ignore").decode().replace("  ", " "))

def normalize():
    origin = open("/Users/Ernesto/PycharmProjects/20NewsGroups/Data.arff", "r")
    dest = open("/Users/Ernesto/PycharmProjects/20NewsGroups/DataN.arff", "w")

    encodetoascii(origin, dest)

    origin.close()
    dest.close()

def loadIMfromtext():
    catfeatures = {}
    fob = open("MIdata.txt", "r")
    reader = csv.reader(fob, delimiter=",")
    cat = ""
    for word, A in reader:
        if word == "Category":
            catfeatures[A] = {}
            cat = A
        else:
            catfeatures[cat][word] = A
    fob.close()
    return catfeatures



vect = importDocs()
#vac = createMutualInformation(vect, categoriesraw)
vac = loadIMfromtext()
#vac = createCollectionFrequency(vect, categoriesraw)
#vac = createDocumentFrequency(vect, categoriesraw)
k = 10000
chosen = chooseVocab(vac, k)
createARFF(vect, chosen, k)
normalize()



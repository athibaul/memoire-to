import numpy as np
import os
import matplotlib.pyplot as plt
import operator
import random
import numpy.linalg
import time

def parse_files(glove='glove.txt',dict_txt='dict.txt'):
    global P, dic, rev_dic
    print("Parsing word embedding...")
    glove_f = open(glove,'r')
    P = [[] for _ in range(100000)]
    for line in glove_f:
        for i,v in enumerate(line.split()):
            P[i].append(float(v))
    P = np.array(P)
    print("Parsing dictionary...")
    dict_f = open(dict_txt,'r')
    dic = dict()
    rev_dic = dict()
    for line in dict_f:
        word,value = line.split()
        dic[word] = int(value)-1
        rev_dic[int(value)-1] = word
    
    print("Done.")
    return P, dic, rev_dic


if 'dic' not in globals():
    parse_files()
    
#useless_words = open('common-english-words.txt','r').read().split(',')
useless_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
noun_list = open('nounlist.txt').read().split()
#print(useless_words)

def build_histogram(file=None):
    if file==None:
        file = random_book()
        print("Working with :",file,"\n\n")
    f = open(file,'r')
    histogram = dict()
    for line in f:
        for word in line.split():
            if word.lower() in noun_list and word.lower() not in useless_words:
                if word in histogram:
                    histogram[word] += 1
                else:  
                    histogram[word] = 1
    return histogram


def random_book():
    l = []
    for path, subdirs, files in os.walk('./discrete_discrete/TEXTSBYAUTHORS/'):
        for name in files:
            l.append(os.path.join(path, name))
    return random.choice(l)

def show_histogram(file=None):
    hist = build_histogram(file)
    sorted_hist = sorted(hist.items(), key=operator.itemgetter(1),reverse=True)
    slice = sorted_hist[:30]
    y_pos = np.arange(len(slice))
    names = [x[0] for x in slice]
    values = [x[1] for x in slice]
    fig,ax = plt.subplots()
    ax.barh(y_pos,values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    plt.show()

def k_closest(v,k=5):
    import heapq
    def key(i_w):
        return numpy.linalg.norm(v-i_w[1])
    return heapq.nsmallest(k,enumerate(P),key=key)

def test_interpolation(word1='video',word2='advertisement'):
    v0 = P[dic[word1],:]
    v1 = P[dic[word2],:]
    
    for t in np.linspace(0,1,20):
        vt = t * v1 + (1-t) * v0
        closest = k_closest(vt)
        print(t,[rev_dic[i] for i,w in closest])
    




def ot_interpolation(files=['./discrete_discrete/TEXTSBYAUTHORS/NAPOLEON/pg3567.txt','./discrete_discrete/TEXTSBYAUTHORS/SHAKESPEARE/shakespeare-romeo-48.txt'], iterations=100, wordcount=3000):
    
    # Build the set of all words found in the books, 
    # and the histogram of each book
    words = set()
    hists = []
    global_hist = dict()
    for book in files:
        print('Reading',book)
        b = open(book,'r')
        histogram = dict()
        for line in b:
            for word in line.split():
                word = word.lower() # Normalize words to all be lowercase
                if word in dic and word.lower() not in useless_words and word.lower() in noun_list:
                    if word in histogram:
                        histogram[word] += 1
                    else:
                        histogram[word] = 1
                    if word in global_hist:
                        global_hist[word] += 1
                    else:
                        global_hist[word] = 1
                    words.add(word)
        hists.append(histogram)
        
    # Take only the 'wordcount' most common words
    sorted_hist = sorted(global_hist.items(), key=operator.itemgetter(1),reverse=True)
    sorted_hist = sorted_hist[:wordcount]
    print('The most common words are :',sorted_hist[:30])
    
    # From now on, a word is referred to by its index in the list of words
    words = [w for w,v in sorted_hist]
    word_index = { w : i for i,w in enumerate(words) }
    n = len(words)
    k = len(files)
    print('n=',n,'; k=',k)
    
    
    # Build the distance matrix
    print("Building distance matrix...")
    M = np.zeros((n,n))
    for i in range(n):
        if 100*i//n < 100*(i+1)//n:
            print('%',end='')
        for j in range(i,n):
            v = P[dic[words[i]]]
            w = P[dic[words[j]]]
            # Norme 2
            M[i,j] = numpy.linalg.norm(v-w)**2
            M[j,i] = M[i,j]
    print("Distance matrix built.")
    print(M)
    
    epsilon = 10
    # -> Verifier violation des contraintes
    # -> Implementation stable ?
    Xi = np.exp(-M/epsilon)
    def xi(v):
        return Xi.dot(v) + 1e-50
    
    
    
    # Build the histogram vectors
    print("Building histogram vectors...")
    c = np.zeros((n,k))
    for i,histogram in enumerate(hists):
        for word,weight in histogram.items():
            try:
                j = word_index[word]
                c[j,i] = weight
            except KeyError:
                pass
        c[:,i] = c[:,i] / np.sum(c[:,i])
    print("Done.")
    
    
    
    if k==2:
        # Apply Sinkhorn once to calculate the distance
        a = np.ones(n); b = np.ones(n)
        err_h = []
        for _ in range(iterations):
            a[:] = c[:,0] / xi(b[:])
            b[:] = c[:,1] / xi(a[:])
            diff = a * xi(b[:]) - c[:,0]
            err = np.sum(np.abs(diff)**2)
            err_h.append(err)
        #plt.semilogy(err_h)
        #plt.show()
        #plt.pause(1)
        dist = (M * Xi).dot(b).dot(a)
        print("Distance is",dist)
    
    def calculate_interpol(lbd=None):
        if lbd is None:
            lbd = np.ones(k)/k
            
        b = np.ones((n,k))
        a = b
        
        for _ in range(iterations):
            for i in range(k):
                a[:,i] = c[:,i] / xi(b[:,i])
            q = np.zeros(n)
            for i in range(k):
                q += lbd[i] * np.log( b[:,i] * xi(a[:,i]) + 1e-50)
            q = np.exp(q)
            for i in range(k):
                b[:,i] = q / xi(a[:,i])
        
        return sorted([(q[i],words[i]) for i in range(n)])
    
    return calculate_interpol



def interpolated_lipsum(ot_interp,total=10000):
    words = [w for v,w in ot_interp]
    values = np.array([v for v,w in ot_interp])
    values *= total/np.sum(values)
    values = np.cumsum(values)
    w_i = 0
    out = ""
    for k in range(total):
        while k > values[w_i]:
            w_i += 1
        out += words[w_i] + " "
    return out

def show_interp(files = ['./discrete_discrete/TEXTSBYAUTHORS/DICKENS/dickens-oliver-627.txt','./discrete_discrete/TEXTSBYAUTHORS/KANT/kant-critique-141.txt'],nb_subplots=5):
    from wordcloud import WordCloud
    interp = ot_interpolation(files)
    def color_func(word,font_size,position,orientation,font_path,random_state):
        return (255,255,255)
    
    print("Calculating interpolations...",end="")
    for i,t in enumerate(np.linspace(0,1,nb_subplots)):
        ot_interp = interp([1-t,t])
        wc = WordCloud(width=300,height=500,color_func=color_func).generate_from_frequencies({ w : f for f,w in ot_interp })
        ax = plt.subplot(1,nb_subplots,i+1)
        ax.imshow(wc,interpolation='bilinear')
        ax.axis('off')
    print(" Done.")
    plt.show()


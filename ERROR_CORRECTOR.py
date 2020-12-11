
import re
import math
import time
import pickle
import operator
import numpy as np
from colorama import init
from termcolor import colored
from prettytable import PrettyTable


init()
print('\n IMPORTING NLTK MODULE...')
t1 = time.time()
import nltk
t2 = time.time()
print('\t[ IMPORTED IN {0:.3f} SECS ]'.format(t2-t1))

file = open( 'NEW_CORPUS.txt' , 'r' )
print(' IMPORTING SAMPLE CORPUS...')
t1 = time.time()
CORPUS = file.read()
file.close()
t2 = time.time()
print('\t[ IMPORTED IN {0:.3f} SECS ]'.format(t2-t1))

print(' LOADING TRIMMED DICTIONARY WORDS...')
TRIMMED = dict()
t1 = time.time()
for i in range(1,16):
    file = open( 'TRIMMED_WORDS_'+str(i), 'rb' )
    TRIMMED.update(pickle.load(file))
    file.close()
t2 = time.time()
print('\t[ LOADED IN {0:.3f} SECS ]'.format(t2-t1))

print(' IMPORTING VOCABULARY...')
t1 = time.time()
VOCAB = list()
file = open( "DICTIONARY.txt" , "r" )
words = list()
for x in file:
    VOCAB.append(x[:-1])
file.close()
t2 = time.time()
print('\t[ IMPORTED IN {0:.3f} SECS ]'.format(t2-t1))
V = len( VOCAB )

print(' LOADING CONFUSION MATRICES...')
t1 = time.time()
file = open( "CONFUSION_MATRIX_DEL.txt" , "r" )
raw = list()
for x in file:
    x = int(x[:-1])
    raw.append(x+1)
file.close()
CMD = np.array(raw).reshape(27,26)

file = open( "CONFUSION_MATRIX_INS.txt" , "r" )
raw = list()
for x in file:
    x = int(x[:-1])
    raw.append(x+1)
file.close()
CMI = np.array(raw).reshape(27,26)

file = open( "CONFUSION_MATRIX_SUB.txt" , "r" )
raw = list()
for x in file:
    x = int(x[:-1])
    raw.append(x+1)
file.close()
CMS = np.array(raw).reshape(27,26)

file = open( "CONFUSION_MATRIX_TRAN.txt" , "r" )
raw = list()
for x in file:
    x = int(x[:-1])
    raw.append(x+1)
file.close()
CMT = np.array(raw).reshape(27,26)
t2 = time.time()
print('\t[ LOADED IN {0:.3f} SECS ]'.format(t2-t1))

print(' LOADING CONTRACTIONS...')
t1 = time.time()
file = open( "WORDS_WITH_APOSTROPHE" , "rb" )
WORDS_W_APOS = pickle.load( file )
file.close()
t2 = time.time()
print('\t[ LOADED IN {0:.3f} SECS ]'.format(t2-t1))

N = 2699132
V = 370104
U = 59613
S = 118697
INFINITY = 10000


def log( x ):
    if not x:
        return -1*float('inf')
    return math.log10(x)


def getEditDistance( x , y ):
    m = len(x)
    n = len(y)
    arr = np.array([0]*((m+1)*(n+1))).reshape( m+1 , n+1 )
    for a in range(m):
        arr[a+1][0] = a+1
    for a in range(n):
        arr[0][a+1] = a+1
    for i in range(1,m+1):
        for j in range(1,n+1):
            v1 = arr[i-1][j] + 1
            v2 = arr[i][j-1] + 1
            v3 = arr[i-1][j-1]
            if x[i-1] != y[j-1]:
                v3 += 1
            v4 = INFINITY
            if ( i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1] ):
                v4 = arr[i-2][j-2] + 1
            v = [ v1 , v2 , v3 , v4 ]
            arr[i][j] = INFINITY
            for each in v:
                if each < arr[i][j]:
                    arr[i][j] = each
    return arr

    
def getCandidates( word , root , s , lev ):
    if ( not len(word) ):
        return
    for x in range(len(word)):
        st = word[:x] + word[x+1:]
        if ( st in TRIMMED.keys() ):
            for each in TRIMMED[st]:
                s.add(each.lower())
        if ( st in VOCAB ):
            s.add(st)
        if ( lev == 1 ):
            getCandidates( st , root , s , 2 )


def getEditOperation( arr , x , y ):
    edits = list()
    i = len(x)
    j = len(y)
    while( i>=0 and j>=0 ):
        if ( not i and not j ):
            break
        p = 0
        if i>0 and j>0 and x[i-1] != y[j-1]:
            p = 1
        if i>0 and j>0 and arr[i][j] == arr[i-1][j-1] + p:
            if p:
                if i == 1:
                    edits.append(('S','#'+x[i-1]+x[i],'#'+y[j-1]+x[i]))
                elif i < len(x):
                    edits.append(('S',x[i-2]+x[i-1]+x[i],x[i-2]+y[j-1]+x[i]))
                else:
                    edits.append(('S',x[i-2]+x[i-1]+'#',x[i-2]+y[j-1]+'#'))
            i = i-1
            j = j-1
            continue
        elif i>0 and arr[i][j] == arr[i-1][j] + 1:
            if i == 1:
                edits.append(('D','#'+x[i-1]+x[i],'#'+x[i]))
            elif i < len(x):
                edits.append(('D',x[i-2]+x[i-1]+x[i],x[i-2]+x[i]))
            else:
                edits.append(('D',x[i-2]+x[i-1]+'#',x[i-2]+'#'))
            i = i-1
            continue
        elif j>0 and arr[i][j] == arr[i][j-1] + 1:
            if not i:
                edits.append(('I','#'+x[i],'#'+y[j-1]+x[i]))
            elif i < len(x) :
                edits.append(('I',x[i-1]+x[i],x[i-1]+y[j-1]+x[i]))
            else:
                edits.append(('I',x[i-1]+'#',x[i-1]+y[j-1]+'#'))
            j = j-1
            continue
        elif ( i > 1 and j > 1 and x[i-1] == y[j-2] and x[i-2] == y[j-1] and arr[i-2][j-2]+1==arr[i][j] ):
                edits.append(('T',x[i-2]+x[i-1],x[i-1]+x[i-2]))
                i = i-2
                j = j-2    
    return edits
 
 
def h( s , axis ):
    if axis == 'h':
        return ord(s[0])-ord('a')
    if s[0] == '#':
        return 0
    return ord(s[0])-ord('a')+1
  
  
def getLikelihood( edits ):
    L = 0
    for edit in edits:
        if edit[0] == 'D':
            if edit[1][0] == '#':
                length = N
            else:
                length = len( re.findall(edit[1][0],CORPUS) )
            L = L + log(CMI[h(edit[1][0],'v')][h(edit[1][1],'h')]) - log(length)
        elif edit[0] == 'I':
            if edit[1][0] == '#':
                length = len( re.findall(' '+edit[2][1],CORPUS) )
            else:
                length = len( re.findall(edit[2][0]+edit[2][1],CORPUS) )
            L = L + log(CMD[h(edit[2][0],'v')][h(edit[2][1],'h')]) - log(length)
        elif edit[0] == 'S':
            length = len( re.findall(edit[2][1],CORPUS) )
            L = L + log(CMS[h(edit[1][1],'v')][h(edit[2][1],'h')]) - log(length)
        elif edit[0] == 'T':
            length = len( re.findall(edit[2][0]+edit[2][1],CORPUS) )
            L = L + log(CMT[h(edit[2][0],'v')][h(edit[2][1],'h')]) - log(length)
    return L


def getUnigramProb( w ):
    if ( w == '' or w == '#' ):
        return log( S + 0.1 ) - log( N + 0.1*V )
    co = len( re.findall( ' ' + w + '[.\s]' , CORPUS ) )
    ca = log( co + 0.1 ) - log( N + 0.1*V )
    return ca


def getBigramProb( w1 , w2 ):
    if ( w1 == '' or w1 == '#' ):
        c12 = len( re.findall( '. ' + w2 + '[.\s]' , CORPUS ) ) + len( re.findall( '^' + w2 + '[.\s]' , CORPUS ) )
        c1 = S
        return log( c12 + math.pow(10,getUnigramProb(w2)) ) - log( c1 + 1 )
    if ( w2 == '' or w2 == '#' ):
        c12 = len( re.findall( ' ' + w1 + '.' , CORPUS ) )
        c1 = len( re.findall( ' ' + w1 + '[.\s]' , CORPUS ) )
        return log( c12 + math.pow(10,getUnigramProb(w2)) ) - log( c1 + 1 )
    c12 = len( re.findall( ' ' + w1 + ' ' + w2 + '[.\s]' , CORPUS ) )
    c1 = len( re.findall( ' ' + w1 + '[.\s]' , CORPUS ) )
    return log( c12 + math.pow(10,getUnigramProb(w2)) ) - log( c1 + 1 )


def getSequenceProb( words , pos ):
    l = len( words )
    if l == 1 and not pos:
        return getBigramProb( '#' , words[pos] ) + getBigramProb( words[pos] , '#' )
    if pos == l-1:
        if words[pos-1] in VOCAB:
            return getBigramProb( words[pos-1] , words[pos] ) + getBigramProb( words[pos] , '#' )
        return getBigramProb( words[pos] , '#' )
    if not pos:
        if words[pos+1] in VOCAB:
            return getBigramProb( '#' , words[pos] ) + getBigramProb( words[pos] , words[pos+1] )
        return getBigramProb( '#' , words[pos] )
    if words[pos-1] in VOCAB:
        if words[pos+1] in VOCAB:
            return getBigramProb( words[pos-1] , words[pos] ) + getBigramProb( words[pos] , words[pos+1] )
        return getBigramProb( words[pos-1] , words[pos] )
    if words[pos+1] in VOCAB:
        return getBigramProb( words[pos] , words[pos+1] )
    return getUnigramProb( words[pos] )
 
 
def bestCandidate( words , pos , choice , CHANGES , sen ):
    ORG = words[pos]
    print('\tNON-WORD ERROR : ' , ORG.upper())
    print('\t     FINDING SUITABLE CANDIDATES...')
    time.sleep(1)
    t1 = time.time()
    cand = set()
    if ( words[pos] in TRIMMED.keys() ):
        cand = cand.union(TRIMMED[words[pos]])
    getCandidates( ORG , ORG , cand , 1 )
    t2 = time.time()
    print( '\t        [ {0} CANDIDATES FOUND IN {1:.3f} SECS ]'.format(len(cand),t2-t1) )
    print( '\t     PROCESSING EACH CANDIDATE...' )
    t1 = time.time() 
    rows = list()
    for c in cand:
        arr = getEditDistance( ORG , c )
        ed = arr[len(ORG)][len(c)]
        if ( ed > 2 ):
            continue
        log_pc = getUnigramProb(c)
        words[pos] = c
        seq_pr = getSequenceProb( words , pos )
        edits = getEditOperation( arr , ORG , c )
        log_likel = getLikelihood( edits )
        rows.append([c, ed, edits, round(-1*log_pc,5), round(-1*seq_pr,5), round(-1*log_likel,5), round(-1*seq_pr-log_likel,5)])
    
    t2 = time.time()
    print( '\t        [ ALL CANDIDATES PROCESSED IN {0:.3f} SECS ]'.format(t2-t1) )
    rows.sort( key = lambda x: (x[1],x[6]) )
    print( '\t        [ {} CANDIDATES SHORT-LISTED ]'.format(len(rows)) )
    words[pos] = rows[0][0]
    CHANGES[(sen,pos)] = ( ORG, words[pos] )
    print( '\t        *[ BEST CANDIDATE : {} ]*'.format(words[pos].upper()) )
    print( '\n' , end = '' )
    if choice == 'y':
        table = PrettyTable(["Correct Word Candidate (C)", "Edit Distance", 
                             "Edit Operations(s) [I->C]", "-log[P(C)]", "-log[P(SEQ)]", "-log[P(I|C)]", "-log[P(SEQ)P(I|C)]"])
        for row in rows:
            table.add_row(row)
        print(table)
        print( '\n' , end='' )


def putApostrophe( w ):
    for pair in WORDS_W_APOS:
        if w == pair[0]:
            return pair[1]
    return w
    

def correctQuery( text , edits ):
    print( '\n ORIGINAL TEXT : ' , colored( text , 'blue' ) )
    query = text.replace('-',' - ').replace("'",'')
    print( '  EDITED TEXT  :' , end = ' ' )
    SENT = nltk.tokenize.sent_tokenize(query)
    ORG_SENT = nltk.tokenize.sent_tokenize(text.replace('-',' - ').replace("'",'-'))
    KEYS = edits.keys()
    
    if not len( KEYS ):
        for s in range(len(ORG_SENT)):
            words = nltk.tokenize.word_tokenize(ORG_SENT[s])
            flag = True
            space = True
            for w in words:
                if w == 'i':
                    w = 'I'
                if flag:
                    w = w.capitalize()
                    flag = not flag
                if re.search( '^[a-zA-Z0-9]+' , w ):
                    if space:
                        print( ' ' , end='' )
                    if not '-' in w:
                        print( colored( putApostrophe(w) , 'green' ) , end = '' )
                    else:
                        if w[:2] == 'i-':   w = 'I' + w[1:]
                        w = w.replace( '-' , "'" )
                        print( colored( w , 'green' ) , end = '' )
                    space = True
                else:
                    if w != '-':
                        flag = True
                        space = True
                    else:
                        space = False
                    print( colored( w , 'green' ) , end = '' )
        return
        
    sentences = set(list(zip(*edits.keys()))[0])
    for s in range(len(SENT)):
        N = -1
        p = -1
        words = nltk.tokenize.word_tokenize(SENT[s])
        ORG_WORDS = nltk.tokenize.word_tokenize(ORG_SENT[s])
        if s+1 in sentences:
            for w in nltk.tokenize.word_tokenize(SENT[s]):
                p = p + 1
                if re.search( '^[a-zA-Z]+$' , w ):
                    N = N + 1
                    if ( s+1 , N ) in KEYS:
                        if words[p][0] >= 'a':
                            words[p] = edits[(s+1,N)][1]
                        else:
                            words[p] = edits[(s+1,N)][1].capitalize()
                    elif '-' in ORG_WORDS[p]:
                        w = ORG_WORDS[p]
                        if ORG_WORDS[p][:2] == 'i-':   w = 'I' + w[1:]
                        words[p] = w.replace( '-' , "'" )
        else:
            for w in nltk.tokenize.word_tokenize(SENT[s]):
                p = p + 1
                if re.search( '^[a-zA-Z]+$' , w ):
                    if '-' in ORG_WORDS[p]:
                        w = ORG_WORDS[p]
                        if ORG_WORDS[p][:2] == 'i-':   w = 'I' + w[1:]
                        words[p] = w.replace( '-' , "'" )
        flag = True
        space = True
        for w in words:
            if w == 'i':
                w = 'I'
            if flag:
                w = w.capitalize()
                flag = not flag
            if re.search( '^[a-zA-Z0-9]+' , w ):
                if space:
                    print( ' ' , end='' )
                if "'" in w:
                    print( colored( w , 'green' ) , end = '' )
                else:
                    print( colored( putApostrophe(w) , 'green' ) , end = '' )
                space = True
            else:
                if w != '-':
                    flag = True
                    space = True
                else:
                    space = False
                print( colored( w , 'green' ) , end = '' )
    

def main( text , choice ):
    query = text.replace('-',' - ').replace("'",'')
    CHANGES = dict()
    counter = 1
    print( '\n' )
    for sentence in nltk.tokenize.sent_tokenize(query):
        flag = True
        print( ' SENTENCE' , counter )
        print( '   SEARCHING NON-WORD ERRORS...' )
        counter = counter + 1
        words = list()
        for w in nltk.tokenize.word_tokenize(sentence):
            if re.search( '^[a-zA-Z]+$' , w ):
                words.append(w.lower())
        for x in range(len(words)):
            if not words[x] in VOCAB:
                bestCandidate( words , x , choice , CHANGES , counter-1 )
                flag = False
        if flag:
            print( '\n' , end='' )
            
    correctQuery( text , CHANGES )
    print('\n' , end='')
  
  
while True:
    print( '\n' + '-'*200 + '\n' )
    query = input( ' ENTER QUERY : ' )
    if not re.search( '[a-zA-Z]' , query ):
        continue
    if ( query[:5].lower() == '#quit' ):
        break
    choice = input( ' SHOW THE CALCULATIONS (Y/N) : ' )
    main( query , choice[0].lower() )


print( colored('\n\n   CLOSING...','magenta') )

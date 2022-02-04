import sys

# argument list:
# - stemmed: true or false
# - representation: BOW, TFIDF, LDA_10, LDA_25, LDA_50, Word2Vec, Glove, Doc2Vec
# - eps: 0.1, 0.25, 0.5
# - min_sample: 5, 10, 20

stemmed = sys.argv[1]
representation = sys.argv[2]
eps = float(sys.argv[3])
min_sample = int(sys.argv[4])

if (stemmed not in ['true', 'false']):
    print('incorrect stemmed argument value: ' + stemmed)
    quit()
    
if (representation not in ['BOW', 'TFIDF', 'LDA_10', 'LDA_25', 'LDA_50', 'Word2Vec', 'Glove', 'Doc2Vec']):
    print('incorrect dbscan representation argument value: ' + representatioon)
    quit()
    
if (eps not in [0.1, 0.25, 0.5]):
    print('incorrect dbscan eps argument value: ' + eps)
    quit()
    
if (min_sample not in [5, 10, 20]):
    print('incorrect dbscan min_sample argument value: ' + min_sample)
    quit()
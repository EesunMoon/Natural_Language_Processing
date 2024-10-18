import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    # get_lexicon(): Instead of pre-defining a lexicon, collect one from the training corpus
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    # (PART 1) extracting n-grams from a sentence
    """
    example)
        >>> get_ngrams(["natural","language","processing"],1)
        [('START',), ('natural',), ('language',), ('processing',), ('STOP',)]
        >>> get_ngrams(["natural","language","processing"],2)
        ('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]
        >>> get_ngrams(["natural","language","processing"],3)
        [('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]
    """

    if len(sequence) == 0 or len(sequence) < n:
        return sequence
    if n == 1:
        sequence = ["START"] + sequence + ["STOP"]
    else:
        sequence = ["START"] * (n-1) + sequence + ["STOP"]

    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # denominator - compute the total number of words: exclude "START" token
        self.denominator = sum(self.unigramcounts.values()) - self.unigramcounts[('START',)]
        # print("denominator:", self.denominator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 

        ##Your code here
        # (PART 2) Counting n-grams in a corpus: count the occurrence frequencies
        """
        >>> model.trigramcounts[('START','START','the')]
        5478
        >>> model.bigramcounts[('START','the')]
        5478
        >>> model.unigramcounts[('the',)]
        61428
        """

        self.num_sentences = 0
        for sentence in corpus:
            self.num_sentences += 1
            get_unigrams = get_ngrams(sentence, 1)
            get_bigrams = get_ngrams(sentence, 2)
            get_trigrams = get_ngrams(sentence, 3)

            # unigrams
            for unigram in get_unigrams:
                self.unigramcounts[unigram] += 1

            #bigrams
            for bigram in get_bigrams:
                self.bigramcounts[bigram] += 1

            # trigram
            for trigram in get_trigrams:
                self.trigramcounts[trigram] += 1    

        # print("unigram", self.unigramcounts)
        # print("bigram", self.bigramcounts)
        # print("trigram", self.trigramcounts)
        return

    """
        One issue you will encounter is the case if which you have a trigram u,w,v   
        where  count(u,w,v) = 0 but count(u,w) is also 0. 
        In that case, it is not immediately clear what P(v | u,w) should be. 
        My recommendation is to make P(v | u,w) = 1 / |V|  (where |V| is the size of the lexicon), if count(u,w) is 0. 
        That is, if the context for a trigram is unseen, the distribution over all possible words in that context is uniform.  
        Another option would be to use the unigram probability for v, so P(v | u,w) = P(v). 
    """

    # (PART 3) Raw n-gram probabilities: unsmoothed probability
    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # p(w|u,v) = count(u, v, w) / count(u, v)

        # P(w|START, START): the number of sentences in the training data as the denominator
        self.V = len(self.lexicon)
        if trigram[:-1] == ('START', 'START'):
            # return self.trigramcounts[trigram] / self.num_sentences
            return 1/self.V
        
        # unseen bigram:: if count(u,w) is 0, then use the unigram probability for v, so P(v | u,w) = P(v)
        if self.bigramcounts[trigram[:2]] == 0:
            return self.unigramcounts[trigram[2:]]/self.denominator # p(v)
        
        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # p(w|u) = count(u,w) / count(u)
        
        
        if self.unigramcounts[bigram[:1]] == 0:
            return 1/self.V
        
        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:1]]
        
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        
        return self.unigramcounts[unigram]/self.denominator

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        """
            linear interpolation
            p(w|u,v) = lambda1 * p_mle(w|u,v)
                        + lambda2 * p_mle(w|v)
                        + lambda3* p_mle(w)

                        input u,v,w
                        # p(w|u,v) = count(u, v, w) / count(u, v)
                        # p(w|v) = count(v, w) / count(v) <- [1:]
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
    
        tri_value = lambda1*self.raw_trigram_probability(trigram)
        bi_value = lambda2*self.raw_bigram_probability(trigram[1:])
        uni_value = lambda3*self.raw_unigram_probability(trigram[2:])
        
        return tri_value + bi_value + uni_value
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        
        """
        # log p(w1, ..., wn) = sum(log p(wi|wi-1))
        
        trigrams = get_ngrams(sentence, 3)  # compute trigrams
        logprob = 0.0                       # Use smoothed_trigram_probability
        for trigram in trigrams:
            smoothed_prob = self.smoothed_trigram_probability(trigram)
            
            # in case of log 0
            if smoothed_prob == 0:
                logprob += 0
            else:
                logprob += math.log2(smoothed_prob)
        
        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        """
            perplexity = 2^(-l)
            l = (1/M) * sum(i=1 to m)log P(s_i)
                M: the total number of word tokens in the test corpus
                m: the number of sentences in the test corpus
                p(s_i): sentence probability

            
            test data: Brown corpus(brown_test.txt)
            perplexity < 400
        """

        total_words = 0 # M
        logprob = 0     # sum(i=1 to m) log p(s_i)
        for sentence in corpus:
            
            # sum(i=1 to m) log p(s_i)
            sent_logprob = self.sentence_logprob(sentence) # s_i
            logprob += sent_logprob
            
            total_words += len(sentence)+1 # words_token + end
        
        l = logprob / total_words

        # print("perplexity:", 2**(-l))
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        # essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")

        model1 = TrigramModel(training_file1) # train_high
        model2 = TrigramModel(training_file2) # train_low

        total = 0
        correct = 0       

        # test_high
        for f in os.listdir(testdir1):
            total += 1
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            
            if pp < pp_model2:
                correct += 1

        # print("testdir1")
        # print("perplexity high:", pp)
        # print("perplexity low:", pp_model2)
        

        # test_low
        for f in os.listdir(testdir2):
            total += 1
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            
            if pp < pp_model1:
                correct += 1
        
        # print("testdir2")
        # print("perplexity high:", pp)
        # print("perplexity low:", pp_model1)
        
        return correct / total

if __name__ == "__main__":
    
    model = TrigramModel(sys.argv[1])
    """
    print("brown train")
    model = TrigramModel("NLP/Assignment1/hw1_data/brown_train.txt") 

    print(model.trigramcounts[('START','START','the')])
    print(model.bigramcounts[('START','the')])
    print(model.unigramcounts[('the',)])
    train_corpus = corpus_reader("NLP/Assignment1/hw1_data/brown_train.txt", model.lexicon) #
    pp = model.perplexity(train_corpus)
    print("perplexity train:", pp)
    # print(model.unigramcounts)
    # print(model.bigramcounts)
    """

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    """
    print("brown test")
    dev_corpus = corpus_reader("NLP/Assignment1/hw1_data/brown_test.txt", model.lexicon) #
    """
    pp = model.perplexity(dev_corpus)
    print("preplexity")
    print(pp)

    # Essay scoring experiment: 
    """
    acc = essay_scoring_experiment('NLP/Assignment1/hw1_data/ets_toefl_data/train_high.txt', 'NLP/Assignment1/hw1_data/ets_toefl_data/train_low.txt', 
                                   "/Users/eesun/CODE/Columbia/Fall2024/NLP/Assignment1/hw1_data/ets_toefl_data/test_high", "/Users/eesun/CODE/Columbia/Fall2024/NLP/Assignment1/hw1_data/ets_toefl_data/test_low")
    """
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")

    print("accuracy")
    print(acc)


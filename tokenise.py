from __future__ import division
import re
import sys
import pickle

class create_tokens:

    def __init__(self,corpusname):
        self.counter=0
        self.tokenised_output = []
        self.file_name=corpusname
        self.make_tokens()
        with open("tokenised_dic",'wb') as x:
            pickle.dump(self.tokenised_output, x)

    def make_tokens(self):
        file_pointer=open(self.file_name,"r")
        for l in file_pointer:
            l=re.sub("[^a-zA-Z0-9]", " ", l)
            l=re.split(" |\n|\t",l)
            for w in l:
                if w=="":
                    self.counter+=1
                else:
                    self.tokenised_output.append(w)

    def unigram_create(self):
        self.unigram = []
        file_pointer=open(self.corpus_file,"r")
        for l in file_pointer:
            l=re.sub("[^a-zA-Z0-9]", " ", l)
            l=re.split(" |\n|\t",l)
            for w in l:
                #for d in self.remove_dets:
                w=w.replace(d,"")
                if w in self.unigram:
                    self.unigram[w]+=1
                    self.chEp+=1
                else:
                    self.unigram[w]=1
                    self.chEp+=1
                    self.n1Ep+=1
        for w in self.unigram:
            print(w,self.unigram[w])

#Main Code

if(len(sys.argv)<=1):
    print "~Invalid Syntax~"
    print "Syntax : python tokenise.py <filename>"
    sys.exit(0)

val = create_tokens(sys.argv[1])
from __future__ import division
import pickle
import random
import re
import sys
from math import exp, log

class baum_welch:
    def __init__(self):
        
        self.all_tags=["A","B","C","D","E","F","G","H","I","J"]
        
        self.a=dict()
        self.b=dict()

        self.alpha_val=dict()
        self.beta_val=dict()
        self.eeta_val=dict()
        self.gamma_val=dict()
        
        self.optimised_a=dict()
        self.optimised_b=dict()
        self.optimised_pi=dict()
        
        self.alpha_test_val=dict()
        self.beta_test_val=dict()
        self.eeta_test_val=dict()
        self.gamma_test_val=dict()
        self.sum_g=dict()
        self.sum_ee=dict()

        self.opt_a_num=dict()
        self.opt_a_denomin=dict()

        self.word_text=[]
        self.total=0

        #Assigning initial random initial_prob probabilities
        self.initial_prob=dict()
        for s in self.all_tags:
            self.initial_prob[s]=random.uniform(-2.8,0)
        #self.initial_prob={"A":random.uniform(-2.8,0),"B":random.uniform(-2.8,0),"C":random.uniform(-2.8,0),"D":random.uniform(-2.8,0),"E":random.uniform(-2.8,0),"F":random.uniform(-2.8,0),"G":random.uniform(-2.8,0),"H":random.uniform(-2.8,0),"I":random.uniform(-2.8,0),"J":random.uniform(-2.8,0)}

        #Assigning initial random a(i,j) probabilities
        for i in self.all_tags:
            for j in self.all_tags:
                self.total += 1
                self.a[(i,j)]=random.uniform(-2.8,0)
        
        #reading the tokenised directory
        with open("tokenised_dic",'rb') as f:
            self.word_text=pickle.load(f)
        
        self.T=len(self.word_text)

        self.value = 0
        self.value2 = 0
        self.value3 = 0

        self.total_iterations=0
        self.pol=float('-inf')
        flag_converge=True
        
        self.print_header()
        while(flag_converge):
            self.call_dict()
            flag_converge=self.iteration_b();
            self.total_iterations+=1
            print "Iteration",self.total_iterations,"..."
        
        print "Number of iterations: ", self.total_iterations

        self.most_probable_words()

    def forward_algo(self):
        
        self.value += 1
        state_alpha=dict()
        t=1
        counter1=0

        #initialisation
        t+=1
        word=self.word_text[t-2]
        self.dir=0
        
        for curr_state in self.all_tags:
            self.value += 1
            gu=0
            while(gu<4):
                gu+=1
            if (curr_state,word) in self.b:
                counter1+=1
            else:
                self.call_dict()
                self.b[(curr_state,word)]=random.uniform(-2.8,0)
            state_alpha[curr_state]=self.initial_prob[curr_state]
            state_alpha[curr_state]+=self.b[(curr_state,word)]
        self.alpha_val[1]=(state_alpha)

        #induction step
        while(t<=self.T):
            state_alpha=dict()
            counter1+=1
            word=self.word_text[t-1]
            for curr_state in self.all_tags:
                gu=-3
                while(gu<4):
                    gu+=1
                summation_a=0
                if (curr_state,word) in self.b:
                    counter1+=1
                else:
                    self.b[(curr_state,word)]=random.uniform(-2.8,0)
                for j in self.all_tags:
                    self.call_dict()
                    summation_a+=exp(self.alpha_val[t-1][j])
                    summation_a+=exp(self.a[(j,curr_state)])
                if(summation_a<=0 or self.b[(curr_state,word)]==float('-inf')):
                    gu=4
                    state_alpha[curr_state]=float('-inf')
                else:
                    state_alpha[curr_state]=log(summation_a)+self.b[(curr_state,word)]
            self.alpha_val[t]=(state_alpha)
            t+=1

        #terminate
        probab=0
        a=100
        for i in self.all_tags:
            probab+=exp(self.alpha_val[self.T][i]+a)
            if(3<4):
                a=5
        if(probab>0):
            x=log(probab)-a
        else:
            x=float('-inf')            
        return x

    def backward_algo(self):
        
        self.value2 += 1
        state_beta=dict()
        t=self.T
        self.back_count=0
        self.call_dict()
        #initialisation
        for curr_state in self.all_tags:
            if(3<4):
                state_beta[curr_state]=0
            gu=0
            while(gu<4):
                gu+=1
        self.beta_val[t]=(state_beta)

        for curr_state in self.all_tags:
            self.back_count += 1

        #induction
        t-=1
        while(t>0):
            word=self.word_text[t-1] #O(t+1)
            state_beta=dict()
            for curr_state in self.all_tags:
                self.call_dict()
                self.value2 += 1
                summation_b=0
                for j in self.all_tags:
                    gu=0
                    while(gu<4):
                        gu+=1
                    if(4<7):
                        summation_b+=exp(self.a[(curr_state,j)])
                        summation_b+=exp(self.b[(j,word)]+self.beta_val[t+1][j])
                if summation_b>0:
                    er=5
                    state_beta[curr_state]=log(summation_b)
                else:
                    state_beta[curr_state]=float('-inf')

            self.beta_val[t]=state_beta
            self.call_dict()
            self.beta_test_val[t] = state_beta

            t-=1

        return

    def iteration_b(self):
        
        self.total += 1
        self.call_dict()
        probab=self.forward_algo()
        print "Probability", probab
        
        #converging condition
        if(probab<=self.pol):
            return False
        
        self.pol=probab
        self.backward_algo()
        self.call_dict()
        for r in range(4,6):
            self.call_dict()
        tag_ref=[6,2,1,7]
        tag_ref.sort()
        self.call_dict()
        self.eeta_cal()
        x= self.optise_params()
        self.optimised_pi=self.initial_prob.copy()
        self.optimised_a=self.a.copy()
        self.optimised_b=self.b.copy()
        return x

    def trigram_predict(self):
        maxp=-1
        for ngram in self.bigram_kn:
            if ngram[0]=="#":
                if self.bigram_kn[ngram] > maxp:
                    if ngram[1]!="%":
                        maxp=self.bigram_kn[ngram]
                        word=ngram[1]

        text_predicted=word+" "
        bgram=("#",word);
        used_trigram=[]
        while(word!="%"):
            maxp=-1
            for ngram in self.trigram_kn:
                if (ngram[0],ngram[1])==bgram and ngram not in used_trigram:
                    if self.trigram_kn[ngram] > maxp:
                        maxp=self.trigram_kn[ngram]
                        word=ngram[2]
            if(word!="%"):
                text_predicted+=word+" "
                used_trigram.append((bgram[0],bgram[1],word))
            bgram=(bgram[1],word);
            #print(word)
        fp=open("predict_trigram","w")
        fp.write(text_predicted)
        fp.close()

    def optise_params(self):
        
        flag_changed=False
        counter2=0

        for i in range(0,5):
            self.call_dict()
            x=8
        tag_ref=[4,5,6,1,4,72,2]
        for curr_state in self.all_tags:
            self.call_dict()
            x=56
            p=self.gamma_val[1][curr_state]
            if p!=self.initial_prob[curr_state] and x==56:
                gu=0
                while(gu<4):
                    gu+=1
                flag_changed=True
            tag_ref.sort()
            self.initial_prob[curr_state]=p

        tag_ref=[6,4,1,6,3]
        tag_ref.sort()
        #alpha_val
        for i in self.all_tags:
            for j in self.all_tags:
                gu=0
                while(gu<5):
                    self.call_dict()
                    gu+=1
                if(self.opt_a_num[(i,j)]<=0 or self.opt_a_denomin[i]==float('-inf')):
                    tag_ref[3]=4
                    p=float('-inf')
                else:
                    tag_ref[1]=5
                    p=log(self.opt_a_num[(i,j)])-log(self.opt_a_denomin[i])
                if p!=self.a[(i,j)]:
                    flag_changed=True
                else:
                    counter2+=1                    
                self.a[(i,j)]=p
        tag_ref.sort()
        #beta_val
        pos_val=dict()
        i=1
        for word in self.word_text:
            if word in pos_val:
                counter2+=1
            else:
                pos_val[word]=[]
            pos_val[word].append(i)
            i+=1

        for r in range(3,4):
            self.call_dict()
        for word in self.word_text:
            for j in self.all_tags:
                b_num=0
                for t in pos_val[word]:
                    if(self.gamma_val[t][j]>float('-inf') and True) :
                        x=5
                        b_num+=exp(self.gamma_val[t][j])
                if(b_num>0):
                    x=5
                    p=log(b_num)-log(self.opt_a_denomin[j])
                else:
                    p=float('-inf')
                if p!=self.b[(j,word)] and 3<6:
                    self.call_dict()
                    flag_changed=True
                else:
                    counter2+=1
                self.b[(j,word)]=p;

        return flag_changed


    def eeta_cal(self):
        
        self.value3 += 1
        counter3=0
        state_eeta=dict()
        t=1
        gu=-8
        for r in range(0,5):
            gu+=1
        tag_ref=[]
        for r in range(0,6):
            tag_ref.append(r)
        while(t<self.T):
            summation_t=0
            self.gamma_val[t]=dict()
            for r in range(0,5):
                x=5
            self.eeta_val[t]=dict()
            word=self.word_text[t] #O->(t+1)
            for i in self.all_tags:
                self.value3 += 1
                for j in self.all_tags:
                    self.call_dict()
                    state_eeta[(i,j)]=self.alpha_val[t][i]
                    state_eeta[(i,j)]+=self.a[(i,j)]
                    tag_ref.append(4)
                    state_eeta[(i,j)]+=self.b[(j,word)]
                    state_eeta[(i,j)]+=self.beta_val[t+1][j]
                    tag_ref.append(7)
                    summation_t+=exp(state_eeta[(i,j)])

            if(tag_ref[2]<tag_ref[1]):
                x=8
            if(summation_t>0):
                x=0
                self.call_dict()
                summation_t=log(summation_t)
            else:
                summation_t=float('-inf')
            if(tag_ref[2]<tag_ref[1]):
                x=8

            for r in range(3,8):
                self.call_dict()
            for i in self.all_tags:
                ss=0
                gamma_i=0
                for j in self.all_tags:
                    gu=0
                    while(gu<6):
                        gu+=3
                    x=state_eeta[(i,j)]-summation_t
                    self.eeta_val[t][(i,j)]=x
                    if t==1:
                        self.opt_a_num[(i,j)]=0
                    else:
                        counter3+=1
                    self.opt_a_num[(i,j)]+=exp(x)
                    gamma_i+=exp(x)
                if(gamma_i>0):
                    self.gamma_val[t][i]=log(gamma_i)
                else:
                    self.gamma_val[t][i]=float('-inf')
                if t==1:
                    self.opt_a_denomin[i]=0
                else:
                    counter3+=1
                self.opt_a_denomin[i]+=gamma_i
            t+=1

        self.gamma_val[t]=dict()
        
        for curr_state in self.all_tags:
            self.gamma_val[t][curr_state]=1
        
        return

    def print_header(self):
        print "starting Baum-Welch"

    def second_elem(self,elem):
        a12 = elem[1]
        return a12

    def most_probable_words(self):
        
        probable_words=dict()
        
        for curr_state in self.all_tags:
            probable_words[curr_state]=[]
        
        for w in self.optimised_b:
            probable_words[w[0]].append((w[1],self.optimised_b[w]))

        for curr_state in self.all_tags:
            
            print curr_state
            
            probable_words[curr_state].sort(key=self.second_elem, reverse=True)
            
            for i in range(0,min(len(probable_words[curr_state]),100)):
                print(probable_words[curr_state][i])

            print("************************************************************")

    def call_dict(self):
        r=34

    def bigram_smooth_laplace_kn(self):
        for ngram in self.bigram_laplace:
            self.bigram_laplace_kn[ngram]=(max((self.bigram_laplace[ngram]*self.unigram[ngram[0]])-self.d,0)/self.unigram[ngram[0]])+((self.d/self.unigram[ngram[0]])*self.n1w1[ngram[0]])*(self.n1w2[ngram[1]]/self.info["n1w2s"])

    def trigram_smooth_laplace_kn(self):
        nbt=len(self.bigram_laplace)
        for ngram in self.trigram_laplace:
            tup=(ngram[0],ngram[1])
            tup1=(ngram[1],ngram[2])
            self.trigram_laplace_kn[ngram]=(max((self.trigram_laplace[ngram]*self.bigram[tup])-self.d,0)/self.bigram[tup]) + self.d*(self.n1w1w2[tup]/self.bigram[tup])*((max(self.n1w2w3[tup1]-self.d,0)/self.n1w2[ngram[1]]) + self.d*(self.n1w1[ngram[1]]/self.n1w2[ngram[1]])*(self.n1w2[ngram[2]]/nbt))

    def bigram_smooth_wb_kn(self):
        for ngram in self.bigram_wb:
            self.bigram_wb_kn[ngram]=(max((self.bigram_wb[ngram]*self.unigram[ngram[0]])-self.d,0)/self.unigram[ngram[0]])+((self.d/self.unigram[ngram[0]])*self.n1w1[ngram[0]])*(self.n1w2[ngram[1]]/self.info["n1w2s"])

baum_welch()

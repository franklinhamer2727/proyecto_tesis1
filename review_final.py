from lib2to3.pgen2.token import STRING
import image
from gensim.models import Word2Vec
import string
import re
import matplotlib
from nltk.corpus import stopwords

with open("/home/software/Escritorio/github.com/simple_scanner_cv/web_analisis/src/resources/skill.txt")  as f:
     content =f.readlines()

content = [x.strip() for x in content]

#visualimos la data
content[1]

import nltk
nltk.download('punkt')
nltk.download('stopwords')


from nltk.tokenize import word_tokenize
import gensim
from gensim.models.phrases import Phraser, Phrases



x = []
for line in content:
    tokens = word_tokenize(line)
    tok =[w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    strpp =[w.translate(table)for w in tok]
    words =[word for word in strpp if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    x.append(words)

texts =x
print(texts[6])
#eliminacion de palbras que se duplican
with open('common.txt') as f:
    content2 =f.read()
ntexts = []
l = len(texts)
for j in range(l):
    s = texts[j]
    res= [i for i in s if i not in content2]
    ntexts.append(res)
print(texts[6])

print(ntexts[6])
print(len(ntexts))
texts =ntexts


content = texts
#Creando bigramas
common_terms = ["of", "with", "without", "and", "or", "the", "a"]
x=ntexts
# Create the relevant phrases from the list of sentences:
phrases = Phrases(sentences =x, connector_words=common_terms)
# The Phraser object is used from now on to transform sentences
bigram = Phraser(phrases)
# Applying the Phraser to transform our sentences
all_sentences = list(bigram[x])
#print(all_sentences)
model=gensim.models.Word2Vec(sentences =all_sentences, vector_size=5000, min_count=2,workers=4,window=4)
print(model)
model.save("final.model")
print(len(model.wv)) #imprimir la longitud de la lista



# testeo
z = model.wv.most_similar("machine_learning")
print(z)
#Analisis

#importando las libririas
import PyPDF2
import os
from os import listdir
from os.path import isfile,join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

#procedemos a la lectura de los cv
mypath =r'/home/software/Escritorio/github.com/simple_scanner_cv/web_analisis/src/static/archivos'
onlyfiles = [os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f))]

#analisis de las palbras de los curriculos
print(onlyfiles)



import collections
def pdfextract(file):
    pdf_file = open(file, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    c = collections.Counter(range(number_of_pages))
    for i in c:
        #page
        page = read_pdf.getPage(i)
        page_content = page.extractText()
    return (page_content.encode('utf-8'))



sim_words=[k[0] for k in model.wv.most_similar("machine_learning")]


def create_bigram(words):
    common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    x=words.split()
# Create the relevant phrases from the list of sentences:
    phrases = Phrases(sentences = x, connector_words=common_terms)
# The Phraser object is used from now on to transform sentences
    bigram = Phraser(phrases)
# Applying the Phraser to transform our sentences is simply
    all_sentences = list(bigram[x])
    
    


def create_profile(file):
    model=Word2Vec.load("final.model")
    text = str(pdfextract(file))
    text = text.replace("\\n", "")
    text = text.lower()
    #print(text)
    #text=create_bigram(text)
    #print(text)
    #below is the csv where we have all the keywords, you can customize your own
    #keyword_dictionary = pd.read_csv(r'C:\Users\dell\Desktop\New folder\ML_CS\NLP\technical_skills.csv')
    stats = [nlp(text[0]) for text in model.wv.most_similar("statistics")]
    NLP = [nlp(text[0]) for text in model.wv.most_similar("language")]
    ML = [nlp(text[0]) for text in model.wv.most_similar("machine_learning")]
    DL = [nlp(text[0]) for text in model.wv.most_similar("deep")]
    #R = [nlp(text) for text in keyword_dictionary['R Language'].dropna(axis = 0)]
    python = [nlp(text[0]) for text in model.wv.most_similar("python")]
    Data_Engineering = [nlp(text[0]) for text in model.wv.most_similar("data")]
    print("*******************************************")
    #print(stats_words,NLP_words)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats)
    matcher.add('NLP', None, *NLP)
    matcher.add('ML', None, *ML)
    matcher.add('DL', None, *DL)
    matcher.add('Python', None, *python)
    matcher.add('DE', None, *Data_Engineering)
    doc = nlp(text)
    
    d = []  
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode I
        span = doc[start : end]               # get the matched slice of the doc
        d.append((rule_id, span.text))      
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    print("KEYWORDS")
    print(keywords)
    
    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    print("********************DF********************")
    print(df)
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
    
       
    name = filename.split('_')
    print(name)
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)
    print("******************DATAF**************")
    print(dataf)

    return dataf




#Code to execute the above functions 
final_db=pd.DataFrame()
i=0
while i < len(onlyfiles):
    file=onlyfiles[i]
    dat=create_profile(file)
    print("************************************* estamos en la linea 204")
    print(type(dat))

    #final_db=final_db.append([dat])
    final_db = pd.concat([final_db,dat])
    i+=1
    print(final_db)



#Code to count words under each category and visualize it through MAtplotlib
final_db2 = final_db['Keyword'].groupby([final_db['Candidate Name'], final_db['Subject']]).count().unstack()
final_db2.reset_index(inplace = True)
final_db2.fillna(0,inplace=True)
candidate_data = final_db2.iloc[:,1:]
candidate_data.index = final_db2['Candidate Name']
#the candidate profile in a csv format
cand=candidate_data.to_csv('candidate_profile.csv')
cand_profile=pd.read_csv('candidate_profile.csv')
cand_profile



import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
# import warnings
# warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
# #
# # Display inline matplotlib plots with IPython
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')


plt.rcParams.update({'font.size': 20})
ax = candidate_data.plot.barh(title="Palabras claves en Currículum según categoría", legend=False, figsize=(25,7), stacked=True)
skills = []
for j in candidate_data.columns:
    for i in candidate_data.index:
        skill = str(j)+": " + str(candidate_data.loc[i][j])
        skills.append(skill)
patches = ax.patches
for skill, rect in zip(skills, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., skill, ha='center', va='center')
plt.show()



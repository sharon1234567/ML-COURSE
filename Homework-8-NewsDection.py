import pandas as pd
import numpy as np
import jieba
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from sklearn.linear_model import LogisticRegression

def token(string):
    return ' '.join(re.findall('[\w|\d]+' , string.replace('\n','')))

def cut(string):
    return list(jieba.cut(string))

def remove_xinhua(string):
    return string.replace('新华社','')
      
def sentence_similarity(s1,s2): 
    return np.dot(s1,s2)/(np.linalg.norm(s1)*np.linalg.norm(s2))

def get_data(df):
    df['sentence vector'] = ''
    for i in range(len(df)):
        string = df.iat[i,0]
        string = remove_xinhua(string)
        string = token(string)
        df.iat[i,0] = string
        df.iat[i,2] = model.infer_vector(cut(string))
    return df
        
def ACC(y,yhat):
    total_wrong = 0
    for i in range(len(y)):
        total_wrong += abs(int(y[i])-int(yhat[i]))
    return 1 - total_wrong/len(y)

database =r'E:\03-NLP课程\素材\export_sql_1558435\sqlResult_1558435.csv'
dataframe = pd.read_csv(database, encoding='gb18030')
dataframe = dataframe.dropna(axis = 0, how = 'any')

all_news = dataframe['content'].tolist()
news_cut = [cut(token(remove_xinhua(n))) for n in all_news]
#model = Word2Vec(news_cut, min_count=1)
#model.save("E:/03-NLP课程/课程作业/Lesson-8/Word2vec.w2v")
#model = gensim.models.Word2Vec.load("E:/03-NLP课程/课程作业/Lesson-8/Word2vec.w2v")

tagged_news = [TaggedDocument(doc, [i]) for i, doc in enumerate(news_cut)]

model = Doc2Vec(tagged_news, vector_size=50, window=2, min_count=1, workers=4)
model.save("E:/03-NLP课程/课程作业/Lesson-8/Doc2vec.d2v")

data_xinhua = dataframe.loc[dataframe['source']=='新华社']
data_others = dataframe.loc[dataframe['source']!='新华社']

data_train_xinhua = pd.DataFrame(data_xinhua[:2000]['content'])
data_test_xinhua = pd.DataFrame(data_xinhua[2001:2500]['content'])
data_train_xinhua['source'] = '1'
data_test_xinhua['source'] = '1'
data_train_others = pd.DataFrame(data_others[:2000]['content'])
data_test_others = pd.DataFrame(data_others[2001:]['content'])
data_train_others['source'] = '0'
data_test_others['source'] = '0'
data_train = data_train_xinhua.append(data_train_others)
data_test = data_test_xinhua.append(data_test_others)

data_train = get_data(data_train)
data_test = get_data(data_test)
    
X_train=[]
X_test=[]

X_train = np.array(data_train['sentence vector'].tolist())   
y_train = data_train['source'].values

X_test = np.array(data_test['sentence vector'].tolist())   
y_test = data_test['source'].values

clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

ACC_train = ACC(y_train, clf.predict(X_train))  # 90.8%
ACC_test = ACC(y_test, clf.predict(X_test))  # 91.5%





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBClassifier
%matplotlib inline

train_df = pd.read_json('./train.json')
test_df = pd.read_json('./test.json')

print "train rows", train_df.shape[0]
print "test rows", test_df.shape[0]

import copy
train_df.drop_duplicates(inplace = True, subset=['listing_id'])
features = list(train_df.columns)
print "train:"
#print "unique listings", train_df.shape[0]
features.remove('created')
#features.remove('display_address')
#features.remove('photos')
#features.remove('features')
features.remove('listing_id')
features.remove('interest_level')
#print features
dedupTrain = copy.deepcopy(train_df)
dedupTrain['photos'] = dedupTrain['photos'].apply(lambda l : ''.join(l))
dedupTrain['features'] = dedupTrain['features'].apply(lambda l : ''.join(l))
dedupTrain.drop_duplicates(subset=features, inplace=True)
print "unique listings", dedupTrain.shape[0]
train_df = train_df.loc[train_df['listing_id'].isin(dedupTrain['listing_id'])]
#train_df = train_df.select(lambda s : not (s['bedroom']==0 and s['bathroom']==0 and s['price']==0), axis=1)
print train_df.shape[0]

#print "unique buildings", train_df.building_id.unique().shape[0]
#print "unique manager", train_df.manager_id.unique().shape[0]

#test_df.drop_duplicates(inplace=True, subset=['listing_id'])
#features.remove('interest_level')
#print "test:"
#print "unique listings", test_df.drop_duplicates(subset=features).shape[0]
#print "unique buildings", test_df.building_id.unique().shape[0]
#print "unique manager", test_df.manager_id.unique().shape[0]

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sets import Set
import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
from collections import defaultdict
from nltk.stem.porter import *
from sklearn.decomposition import TruncatedSVD
from xgboost.sklearn import XGBClassifier
feaDict = {'elevator':['elevator'],
               'hardwood':['hardwoord'],
               'allowed':['allowed'],
               'doorman':['doorman'],
               'dishwasher':['dishwasher'],
               'fee':['fee'],
               'laundry':['laundry', 'dryer','washer'],
               'fitness':['fitness','gym'],
               'roof':['roof'],
               'outdoor':['outdoor'],
               'hardwood':['hardwood'],
               'pool':['pool'],
               'internet':['internet','wifi'],
               'park':['park'],
               'ceiling':['ceiling'],
               'balcony':['balcony','terrace'],
               'patio':['patio','garden','courtyard','backyard'],
               'renovate':['renovate'],
               'prewar':['prewar','pre-war'],
               'garage':['garage']
              }
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key=[]):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class featureCountVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, features, y=None):
        feaDict = defaultdict(int)
        for feas in features:
            for fea in feas.split(','):
                feaDict[fea] += 1
        feaList = sorted(list(feaDict.items()), key=lambda x : x[1], reverse=True)[:300]
        vocab = dict(zip([x[0] for x in feaList], range(len(feaList))))      
        self.vectorizer = CountVectorizer(vocabulary=vocab, binary=True)
        self.vectorizer.fit(features)
        return self
    def transform(self, features):
        return self.vectorizer.transform(features)
    
class preprocessing(BaseEstimator, TransformerMixin):
    def feaCateProcess(self, feaList):
        newFealist = set([])
        for fea in feaList:
            find = False
            for key in feaDict:
                if sum([1 for item in feaDict[key] if item in fea.lower()]) > 0:
                    newFealist.add(key)
                    find = True
                    break
            if not find:
                newFealist.add(fea.lower())
        return list(newFealist)
    def textFea(self, s):
        stop = stopwords.words('english')
        punctuation = set(string.punctuation)
        stemmer = PorterStemmer()
        s = ''.join([c for c in s.lower() if c not in punctuation])
        s = re.sub('[^\w\s]',' ', s) #removes punctuations
        s = re.sub('\d+',' ', s) #removes digits
        s = ' '.join([w for w in s.split() if not w in stop]) # removes english stopwords
        s = ' '.join([w for w , pos in pos_tag(s.split()) if (pos == 'NN' or pos=='JJ' or pos=='JJR' or pos=='JJS' )])
        #selecting only nouns and adjectives
        s = ' '.join([stemmer.stem(w) for w in s.split() if not len(w)<=2 ]) #removes single lettered words and digits
        s = s.strip()
        return s
    def extractFeaFromDesc(self, s, feaMap):
        addFeas = []
        for key in feaMap:
            if ' ' + key + ' ' in s.lower():  #the features are actually in description
                addFeas.append(feaMap[key])
        return addFeas
    def concateTwoFeas(self, serie):
        s = set(serie['featuresFromDescp'])
        return s.union(set(serie['features']))
    def getFeaMap(self, df):
        feaMap = {}
        for key in feaDict:
            for item in feaDict[key]:
                feaMap[item] = key
        for feaList in df['features']:
            for fea in feaList:
                if fea not in feaMap and len(fea)>1 and not fea.isdigit():
                    feaMap[fea]=fea
        return feaMap
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        df['features'] = df['features'].apply(self.feaCateProcess)
        #df['features'] = df['features'].apply(lambda l : ','.join(l))
        df['num_photos'] = df['photos'].apply(len)
        #df['num_features'] = df['features'].apply(len)
        #df['desc_wc'] = df['description'].apply(lambda s : len(s.split()))
        df['cleanDescription'] = df['description'].apply(self.textFea)
        feaMap = self.getFeaMap(df)
        df['featuresFromDescp'] = df['description'].apply(self.extractFeaFromDesc, args=(feaMap,))
        df['combinedFeatures'] = df.apply(self.concateTwoFeas, axis=1)
        df['combinedFeatures'] = df['combinedFeatures'].apply(lambda l : ','.join(l))
        return df
    
pipeline = Pipeline([
    ('preprocess', preprocessing()),
    ('union', FeatureUnion(
        transformer_list=[
            ('numerical', Pipeline([
                ('selector', ItemSelector(key=['bathrooms','bedrooms','price',\
                                               'num_photos','latitude','longitude','building_id','manager_id'])),
            ])),

            ('featureVector', Pipeline([
                ('selector', ItemSelector(key='combinedFeatures')),
                ('featureVector', featureCountVectorizer()),
                ('SVD', TruncatedSVD(n_components=50)),
            ])),

            ('descriptionVector', Pipeline([
                ('selector', ItemSelector(key='cleanDescription')),
                ('tfidf', TfidfVectorizer(max_features=500)),
                ('SVD', TruncatedSVD(n_components=50)),
            ])),

        ],
        # weight components in FeatureUnion
        transformer_weights={
            'numerical': 1,
            'featureVector': 1,
            'descriptionVector': 0.3,
        },
    )),

    ('XGB', XGBClassifier(learning_rate=0.2)),
])
parameters = {
    #'union__numerical__selector__key': [['bathrooms','bedrooms','price','num_photos','latitude','longitude']],
    'union__descriptionVector__tfidf__max_features': [200],
    'XGB__learning_rate': [0.2],
    'XGB__objective': ['multi:softprob'],
    #'XGB__eval_metric': ['mlogloss'],
}
#X = pipeline.fit_transform(train_df) 
#pipeline.fit(train_df, train_df['interest_level'])

clf = GridSearchCV(pipeline, parameters, cv=3, scoring='neg_log_loss')
clf.fit(train_df, train_df['interest_level'])
#print "best para:", clf.best_params_
print "mean validation score:", clf.cv_results_['mean_test_score']
print "validation score std", clf.cv_results_['std_test_score']

clf.best_estimator_.fit(train_df, train_df['interest_level'])  #fit all data
test_predict = clf.predict_proba(test_df)

labels2idx = {label: i for i, label in enumerate(clf.best_estimator_.classes_)}
print labels2idx
sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = test_predict[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)
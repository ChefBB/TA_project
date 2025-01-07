#BASIC ORIGINAL DATASET PREPROCESSING AND SAMPLING 

import pandas as pd


path = '.\\models\\data\\'
df = pd.read_csv(path + 'en_lyrics.csv')

#dropping useless and redundant columns
df = df.drop(
     columns= [
         'id',
         'language_cld3', 'language_ft', 'language'
     ]
)


df['tag'] = df[
     [
         'is_country', 'is_pop', 'is_rap',
         'is_rb', 'is_rock',
     ]
 ].idxmax(axis=1)


#sampling 
df = df.groupby('tag').sample(frac=0.05, random_state=42)


# remove misc values, write df
df = df.loc[df['tag'] != 'misc']


# explode column tag into is_<genre>
df = pd.concat(
     [
         df,
         pd.get_dummies(df['tag'], prefix='is', dtype=bool)
     ], axis=1
 ).drop(columns= 'tag')


# save sampled dataset 
# df.to_csv(path + 'sampled_dataset.csv', index= False)





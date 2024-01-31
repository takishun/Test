#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 共通事前処理

# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

# 必要ライブラリのimport
import pandas as pd
import numpy as np

# データフレーム表示用関数
from IPython.display import display

# 表示オプション調整
# pandasでの浮動小数点の表示精度
pd.options.display.float_format = '{:.4f}'.format

# データフレームですべての項目を表示
pd.set_option("display.max_columns",None)

# ライブラリの読み込み
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



# In[2]:


url = 'https://raw.githubusercontent.com/makaishi2/sample-data/master/data/retail-france.csv'
df = pd.read_csv(url)
display(df[100:110])


# In[3]:


# 発注番号と商品番号をキーに商品個数を集計する
w1 = df.groupby(['発注番号', '商品番号'])['商品個数'].sum()

# 結果確認
print(w1.head())


# In[4]:


# 商品番号を列に移動 (unstack関数の利用)
w2 = w1.unstack().reset_index().fillna(0).set_index('発注番号')

# サイズ確認
print(w2.shape)

# 結果確認
display(w2.head())


# In[5]:


# 集計結果が正か0かでTrue/Falseを設定
basket_df = w2.apply(lambda x: x>0)

# 結果確認
display(basket_df.head())


# In[6]:


# アプリオリによる分析
freq_items1 = apriori(basket_df, min_support = 0.06, 
    use_colnames = True)

# 結果確認
display(freq_items1.sort_values('support', 
    ascending = False).head(10))

# itemset数確認
print(freq_items1.shape[0])


# In[7]:


# アソシエーションルールの抽出
a_rules1 = association_rules(freq_items1, metric = "lift",
    min_threshold = 1)

# リフト値でソート
a_rules1 = a_rules1.sort_values('lift',
    ascending = False).reset_index(drop=True)

# 結果確認
display(a_rules1.head(10))

# ルール数確認
print(a_rules1.shape[0])


# In[ ]:






# coding: utf-8

# ## Overview
# 
# This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# 
# This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# In[2]:


# File location and type
file_location = "/FileStore/tables/*.parquet"
#file_location = "FileStore/tables/part_r_00000_fbc86a65_ecb8_4e20_9ce2_942c31dae5f5_gz-f1668.parquet/"
file_type = "parquet"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)

display(df)


# In[3]:


df.count()


# ## Clean Command

# In[5]:


import re
def clean(yy):
  clean = re.sub(r'[^a-zA-Z\.\s]+',"",yy)
  #clean = re.sub('[\s]'," ",clean)
  return clean
                 
text = "this is test 123''//\\."
clean(text)
                 


# ## Word2Vec

# In[7]:


from pyspark.ml.feature import Word2Vec
word2Vec = Word2Vec(vectorSize=100, seed=42, inputCol="body_cleaned", outputCol="model")
model = word2Vec.fit(df)
#model.getVectors().show()
a = model.getVectors()
a.show()


# In[8]:


b = a.select("vector")
b.show()


# ## Synonyms

# In[10]:


from pyspark.sql.functions import format_number as fmt
m = model.findSynonyms("clarissa", 10).select("word", fmt("similarity", 6).alias("similarity"))
display(m)


# ## PCA

# In[12]:


from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = b
#df = spark.createDataFrame(data, ["features"])

pca = PCA(k=100, inputCol="vector", outputCol="pcaFeatures")
p_model = pca.fit(data)

result = p_model.transform(data).select("pcaFeatures")
result.show(truncate=False)
result.createOrReplaceTempView("result")


# In[13]:


c = result.rdd.map(lambda x:x[0]).collect()
print(c[0])


# In[14]:


print(c)


# ## Bokeh Plot

# In[16]:


#c = result.rdd.map(lambda x:x[0]).collect()
lista = []
for i in c:
  lista.append(i[0])

listb = []
for i in c:
  listb.append(i[1])
  
listc = []
for i in range(len(lista)):
  j = [lista[i],listb[i]]
  listc.append(j)

print(listc)


# In[17]:


d = a.select('word')
display(d)
d.count()


# In[18]:


e = d.rdd.map(lambda x:x).collect()
listd =[]
for g in e:
  listd.append(g[0])
print(listd[:10])


# In[19]:


clean_d = [clean(i) for i in listd]
print(clean_d[:10])


# In[20]:


from bokeh.plotting import figure, output_file, show
from bokeh.embed import components, file_html
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label

x = lista
y = listb
f = clean_d

source = ColumnDataSource(data=dict(x=x, y=y, f=f))


p = figure(plot_width=1000, plot_height=800)

p.scatter(x='x', y='y', size=8, source=source)
#p.circle(x, y,color="navy", alpha=4)
labels = LabelSet(x='x', y='y', text='f', source=source)

p.add_layout(labels)
html = file_html(p, CDN, "my plot1")

displayHTML(html)


# In[21]:


g = a.toPandas()
print(g)


# In[22]:


'''
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.linalg import Vectors

# Load and parse the data
LDAdata = spark.read.csv('/FileStore/tables/export.csv')
#d.createOrReplaceTempView("d")
parsedData = LDAdata.rdd.map(lambda line: Vectors.dense([float(h) for h in line.strip()])) #.split(' ')
# Index documents with unique IDs
corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3)

# Output topics. Each is a distribution over words (matching word count vectors)
print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize())
      + " words):")
topics = ldaModel.topicsMatrix()
for topic in range(3):
    print("Topic " + str(topic) + ":")
    for word in range(0, ldaModel.vocabSize()):
        print(" " + str(topics[word][topic]))
'''


# In[23]:


print(listd[:10])


# In[24]:


from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec

sc = SparkContext.getOrCreate()
#d.createOrReplaceTempView('d')

#k = 100
#w2v = Word2Vec()#.setVectorSize(k)

#inp = sc.textFile("/FileStore/tables/export.txt").map(lambda row: row.split(" "))


#ana = w2v.fit(inp)                                  

def getAnalogy(s, model):
    qry = model(s[0]) - model(s[1]) - model(s[2])
    res = model.findSynonyms((-1)*qry,5) # return 5 "synonyms"
    res = [x[0] for x in res]
    for k in range(0,3):
        if s[k] in res:
            res.remove(s[k])
    return res[0]


# In[25]:


s = ('word', 'cherylene', 'clarissa')
getAnalogy(s, model)


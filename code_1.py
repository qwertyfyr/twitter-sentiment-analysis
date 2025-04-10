import kagglehub
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import col, lower, regexp_replace,regexp_extract, when, to_timestamp, to_date
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("Test").config("spark.driver.memory", "4g").getOrCreate()

path = kagglehub.dataset_download("kazanova/sentiment140")

# Creating schema structure as its not included in the dataset
schemaStructure = StructType([
    StructField("target", IntegerType(), True),
    StructField("id", StringType(), True),
    StructField("date", StringType(), True),
    StructField("flag", StringType(), True),
    StructField("user", StringType(), True),
    StructField("text", StringType(), True)
])

df = spark.read.csv(path, schema=schemaStructure, header=False) #reading csv and creating DataFrame

# Cleaning Tweets
df_clean = (
    df.withColumn("clean_text", lower(col("text")))
    .withColumn("clean_text", regexp_replace("clean_text", r"http\S+|www\S+", "")) # remove URLs
    .withColumn("clean_text", regexp_replace("clean_text", r"@\w+", "")) # Remove mentions
    .withColumn("clean_text", regexp_replace("clean_text", r"[^\w\s]", "")) # Remove punctuation
    .withColumn("clean_text", regexp_replace("clean_text", r"\s+", " ")) # Whitespace being 1
    .withColumn("clean_text", regexp_replace("clean_text", r"#\w+", " ")) # Whitespace being 1
    .withColumn("clean_date", regexp_replace("date", r"^[A-Za-z]{3}\s|\d{2}:\d{2}:\d{2}\s\w{3}\s", ""))
)

df_clean.select("date", "clean_date").show(5, truncate=False)

# Tokenizing
tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="tokens", pattern="\\s+")
df_tokenized = tokenizer.transform(df_clean)

# Remove stopwords
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
df_filtered = remover.transform(df_tokenized)

df_filtered.select("filtered_tokens").show(5, truncate=False)


# Training the Classifier
df_labeled = df_filtered.withColumn("label", when(col("target") == 4, 2) \
                                                .when(col("target") == 2, 1) \
                                                .otherwise(0))

word2vec = Word2Vec(vectorSize=100, inputCol="filtered_tokens", outputCol="features")

# Classifier
lr = LogisticRegression(maxIter=10, regParam=0.01)

pipeline = Pipeline(stages=[word2vec, lr])
model = pipeline.fit(df_labeled)

# Predict
predictions = model.transform(df_labeled)
predictions.select("label", "prediction").show(5)


# Sentiment Trends
df_trend = predictions.withColumn("date_only", to_date("clean_date", "MMM dd yyyy"))

daily_sentiment = df_trend.groupBy("date_only", "prediction").count().orderBy("date_only")
daily_sentiment.show()

df_with_hashtags = predictions.withColumn("hashtag", regexp_extract("text", r"#(\w+)", 1))

# filter out rows not containing hashtags
df_with_hashtags = df_with_hashtags.filter(col("hashtag") != "")


# Group by hashtag and sentiment
hashtag_trend = df_with_hashtags.groupBy("hashtag", "prediction").count().orderBy("count", ascending=False)
hashtag_trend.show(5)
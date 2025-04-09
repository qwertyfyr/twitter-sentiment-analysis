import kagglehub
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.functions import udf, col, lower, regexp_replace 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer

spark = SparkSession.builder.appName("Test").getOrCreate()

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
    df.withColumn("text", lower(col("text")))
    .withColumn("text", regexp_replace("text", r"http\S+|www\S+", "")) # remove URLs
    .withColumn("text", regexp_replace("text", r"@\w+", "")) #Remove mentions
    .withColumn("text", regexp_replace("text", r"#\w+", "")) #Remove hashtag symbol
    .withColumn("text", regexp_replace("text", r"[^\w\s]", "")) #Remove punctuation
    .withColumn("text", regexp_replace("text", r"\s+", " ")) # Whitespace being 1
)


# Tokenizing
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")

df_tokenized = tokenizer.transform(df_clean)
df_tokenized.show()

# Classification
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

#fit the model fails here.
lrModel = lr.fit(df_tokenized)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


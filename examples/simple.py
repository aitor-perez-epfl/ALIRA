from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from alira import ActiveLearner

# Read dataset with movie texts
# (built from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
movies = pd.read_csv('movies.csv')

learner = ActiveLearner()

results_df, _, _ = learner.fit(df=movies, query="superheros")

print(results_df)

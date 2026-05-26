from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from alira import ActiveLearner

# Read dataset with movie texts
# (built from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
movies = pd.read_csv('data/movies.csv')

learner = ActiveLearner()
superhero_movies, _, _ = learner.fit(df=movies, query="superheroes")

print(superhero_movies[['text', 'score']])

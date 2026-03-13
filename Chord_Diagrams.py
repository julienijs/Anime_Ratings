import pandas as pd
import itertools
from collections import Counter
import holoviews as hv
from holoviews import opts

hv.extension('bokeh')

# Load data
df = pd.read_csv("Anime_Data/tv_anime_ratings.csv")

# Drop rows with empty genres
df = df.dropna(subset=['genres'])

# Split genres and create pairs
genre_pairs = []
for genres in df['genres']:
    genre_list = [g.strip() for g in genres.split(',')]
    genre_pairs.extend(itertools.combinations(sorted(genre_list), 2))

# Count co-occurrences
pair_counts = Counter(genre_pairs)

# Prepare data for chord
chord_data = [(a, b, count) for (a, b), count in pair_counts.items()]

# Create Chord diagram
chord = hv.Chord(chord_data)
chord.opts(
    opts.Chord(
        labels='index',
        edge_color='source',
        node_color='index',
        cmap='Category20',
        edge_cmap='Category20',
        width=600, height=600
    )
)

hv.save(chord, 'chord.html')
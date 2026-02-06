import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("Anime_Data/tv_anime_ratings.csv")

# --------------------------------------------------
# Filter: 2014 anime with score >= 6
# --------------------------------------------------
df = df[
    (df["year"] == 2014) &
    (df["score"] >= 6)
].copy()

# Handle missing demographics
df["demographic"] = df["demographics"].fillna("Unknown")

# --------------------------------------------------
# Marker mapping by demographic
# --------------------------------------------------
marker_map = {
    "Shounen": "o",   # circle
    "Seinen": "s",    # square
    "Shoujo": "^",    # triangle
    "Josei": "D",     # diamond
    "Kids": "P",      # plus-filled
    "Unknown": "X"
}

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(10, 6))

for demographic, group in df.groupby("demographic"):
    # Use first demographic if multiple are listed
    primary_demo = demographic.split(",")[0]
    marker = marker_map.get(primary_demo, "X")

    sizes = (group["score"] ** 2) * 20  # scale size by score

    plt.scatter(
        group["score"],
        group["members"],
        s=sizes,
        marker=marker,
        label=primary_demo,
        alpha=0.7
    )

plt.xlabel("MyAnimeList Score")
plt.ylabel("Members (Popularity)")
plt.title(
    "TV Anime from 2014 (Score ≥ 6)\n"
    "Shape = Demographic | Size = Score"
)

plt.legend(title="Demographic")
plt.tight_layout()
plt.show()

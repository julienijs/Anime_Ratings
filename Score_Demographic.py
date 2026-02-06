import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Create output directory
# --------------------------------------------------
os.makedirs("Visualizations", exist_ok=True)

# --------------------------------------------------
# Load and filter data
# --------------------------------------------------
df = pd.read_csv("Anime_Data/tv_anime_ratings.csv")

# Keep valid years and scores
df = df.dropna(subset=["year", "score"])
df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]

# --------------------------------------------------
# Demographic processing
# --------------------------------------------------
df["demographics"] = df["demographics"].fillna("")
df = df.assign(
    demographic=df["demographics"].str.split(", ")
).explode("demographic")

valid_demographics = ["Shoujo", "Shounen", "Josei", "Seinen"]
df = df[df["demographic"].isin(valid_demographics)]

# --------------------------------------------------
# Color palette (consistent & editable)
# --------------------------------------------------
colors = {
    "Shoujo": "#ffb3c6",
    "Shounen": "#00b4d8",
    "Josei": "#da2c43",
    "Seinen": "#03045e"
}

order = ["Shounen", "Shoujo", "Seinen", "Josei"]

# ==================================================
# 1. Mean score over time by demographic
# ==================================================
mean_scores = (
    df.groupby(["year", "demographic"])["score"]
      .mean()
      .unstack()
      .reindex(columns=order)
)

plt.figure(figsize=(12, 7))
for demo in order:
    plt.plot(
        mean_scores.index,
        mean_scores[demo],
        label=demo,
        color=colors[demo],
        linewidth=2
    )

plt.xlabel("Year")
plt.ylabel("Mean Score")
plt.title("Mean TV Anime Score Over Time by Demographic (1990–2025)")
plt.legend(
    title="Demographic",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("Visualizations/score_trends_by_demographic.png", dpi=300)
plt.close()

# ==================================================
# 2. Score distribution by demographic (boxplots)
# ==================================================
plt.figure(figsize=(10, 7))

data = [df[df["demographic"] == d]["score"] for d in order]

box = plt.boxplot(
    data,
    tick_labels=order,
    patch_artist=True,
    showfliers=False
)

for patch, demo in zip(box["boxes"], order):
    patch.set_facecolor(colors[demo])
    patch.set_alpha(0.85)

plt.xlabel("Demographic")
plt.ylabel("Score")
plt.title("Score Distribution by Demographic (1990–2025)")
plt.tight_layout()
plt.savefig("Visualizations/score_distribution_by_demographic.png", dpi=300)
plt.close()


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
demo_colors = {
    "Shoujo": "#ffb3c6",
    "Shounen": "#00b4d8",
    "Josei": "#da2c43",
    "Seinen": "#03045e"
}

demo_order = ["Shoujo", "Shounen", "Josei", "Seinen"]

# ==================================================
# 1. Mean score over time by demographic
# ==================================================
mean_scores = (
    df.groupby(["year", "demographic"])["score"]
      .mean()
      .unstack()
      .reindex(columns=demo_order)
)

plt.figure(figsize=(12, 7))
for demo in demo_order:
    plt.plot(
        mean_scores.index,
        mean_scores[demo],
        label=demo,
        color=demo_colors[demo],
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

data = [df[df["demographic"] == d]["score"] for d in demo_order]

box = plt.boxplot(
    data,
    labels=demo_order,
    patch_artist=True,
    showfliers=False
)

for patch, demo in zip(box["boxes"], demo_order):
    patch.set_facecolor(demo_colors[demo])
    patch.set_alpha(0.85)

plt.xlabel("Demographic")
plt.ylabel("Score")
plt.title("Score Distribution by Demographic (1990–2025)")
plt.tight_layout()
plt.savefig("Visualizations/score_distribution_by_demographic.png", dpi=300)
plt.close()

# ==================================================
# 3. Mean score by demographic (overall)
# ==================================================
overall_means = (
    df.groupby("demographic")["score"]
      .mean()
      .reindex(demo_order)
)

plt.figure(figsize=(8, 6))
plt.bar(
    overall_means.index,
    overall_means.values,
    color=[demo_colors[d] for d in overall_means.index]
)

plt.xlabel("Demographic")
plt.ylabel("Mean Score")
plt.title("Overall Mean TV Anime Score by Demographic (1990–2025)")

for i, value in enumerate(overall_means.values):
    plt.text(i, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("Visualizations/overall_mean_score_by_demographic.png", dpi=300)
plt.close()

print("Score vs Demographic analysis completed.")

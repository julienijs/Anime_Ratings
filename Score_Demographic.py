import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

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

# Set Shounen as reference category
df["demographic"] = pd.Categorical(
    df["demographic"],
    categories=["Shounen", "Shoujo", "Seinen", "Josei"],
    ordered=False
)

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

# ==================================================
# 3. LOESS smoothing of score trends
# ==================================================
plt.figure(figsize=(12, 7))

for demo in order:
    subset = df[df["demographic"] == demo]
    yearly = subset.groupby("year")["score"].mean().reset_index()

    smoothed = lowess(
        yearly["score"],
        yearly["year"],
        frac=0.15
    )

    plt.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        label=demo,
        color=colors[demo],
        linewidth=2
    )

plt.xlabel("Year")
plt.ylabel("Smoothed Mean Score")
plt.title("Smoothed Score Trends by Demographic (LOESS)")
plt.legend(
    title="Demographic",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("Visualizations/score_trends_loess_by_demographic.png", dpi=300)
plt.close()

# ==================================================
# 4. Regression: score ~ year * demographic
# ==================================================
print("\n==============================")
print("OLS Regression: score ~ year * demographic")
print("==============================\n")

ols_model = smf.ols(
    "score ~ year * demographic",
    data=df
).fit()

print(ols_model.summary())

# ==================================================
# 5. Mixed-effects model: random intercept by genre
# ==================================================
df_me = df.copy()
df_me["genres"] = df_me["genres"].fillna("")
df_me = df_me.assign(
    genre=df_me["genres"].str.split(", ")
).explode("genre")
df_me = df_me[df_me["genre"] != ""]

print("\n==============================")
print("Mixed-Effects Model: score ~ year + demographic + (1 | genre)")
print("==============================\n")

mixed_model = smf.mixedlm(
    "score ~ year + demographic",
    data=df_me,
    groups=df_me["genre"]
).fit(reml=False)

print(mixed_model.summary())

# ==================================================
# 6. Score × genre × demographic interaction
# ==================================================
TOP_GENRES_INTERACTION = 5
top_genres_int = df_me["genre"].value_counts().head(TOP_GENRES_INTERACTION).index
df_int = df_me[df_me["genre"].isin(top_genres_int)]

print("\n==============================================")
print("OLS Interaction Model: score ~ year * genre * demographic")
print("==============================================\n")

interaction_model = smf.ols(
    "score ~ year * genre * demographic",
    data=df_int
).fit()

print(interaction_model.summary())

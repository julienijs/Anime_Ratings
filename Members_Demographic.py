import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
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

# Keep valid years and members
df = df.dropna(subset=["year", "members"])
df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]

# Log-transform members for modeling
df["log_members"] = np.log1p(df["members"])

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
# Color palette
# --------------------------------------------------
colors = {
    "Shoujo": "#ffb3c6",
    "Shounen": "#00b4d8",
    "Josei": "#da2c43",
    "Seinen": "#03045e"
}

order = ["Shounen", "Shoujo", "Seinen", "Josei"]

# ==================================================
# 1. Mean members over time by demographic
# ==================================================
mean_members = (
    df.groupby(["year", "demographic"])["members"]
      .mean()
      .unstack()
      .reindex(columns=order)
)

plt.figure(figsize=(12, 7))
for demo in order:
    plt.plot(
        mean_members.index,
        mean_members[demo],
        label=demo,
        color=colors[demo],
        linewidth=2
    )

plt.xlabel("Year")
plt.ylabel("Mean Members")
plt.title("Mean TV Anime Popularity (Members) Over Time by Demographic (1990–2025)")
plt.legend(
    title="Demographic",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("Visualizations/members_trends_by_demographic.png", dpi=300)
plt.close()

# ==================================================
# 2. Members distribution by demographic (boxplots)
# ==================================================
plt.figure(figsize=(10, 7))

data = [df[df["demographic"] == d]["members"] for d in order]

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
plt.ylabel("Members")
plt.title("Popularity Distribution (Members) by Demographic (1990–2025)")
plt.tight_layout()
plt.savefig("Visualizations/members_distribution_by_demographic.png", dpi=300)
plt.close()

# ==================================================
# 3. LOESS smoothing of members trends
# ==================================================
plt.figure(figsize=(12, 7))

for demo in order:
    subset = df[df["demographic"] == demo]
    yearly = subset.groupby("year")["members"].mean().reset_index()

    smoothed = lowess(
        yearly["members"],
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
plt.ylabel("Smoothed Mean Members")
plt.title("Smoothed Popularity Trends by Demographic (LOESS)")
plt.legend(
    title="Demographic",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("Visualizations/members_trends_loess_by_demographic.png", dpi=300)
plt.close()

# ==================================================
# 4. Regression: log(members) ~ year * demographic
# ==================================================
print("\n==============================")
print("OLS Regression: log(members) ~ year * demographic")
print("==============================\n")

ols_model = smf.ols(
    "log_members ~ year * demographic",
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
print("Mixed-Effects Model: log(members) ~ year + demographic + (1 | genre)")
print("==============================\n")

mixed_model = smf.mixedlm(
    "log_members ~ year + demographic",
    data=df_me,
    groups=df_me["genre"]
).fit(reml=False)

print(mixed_model.summary())

# ==================================================
# 6. Members × genre × demographic interaction
# ==================================================
TOP_GENRES_INTERACTION = 5
top_genres_int = df_me["genre"].value_counts().head(TOP_GENRES_INTERACTION).index
df_int = df_me[df_me["genre"].isin(top_genres_int)]

print("\n===================================================")
print("OLS Interaction Model: log(members) ~ year * genre * demographic")
print("===================================================\n")

interaction_model = smf.ols(
    "log_members ~ year * genre * demographic",
    data=df_int
).fit()

print(interaction_model.summary())

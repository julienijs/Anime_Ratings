import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.formula.api as smf

# --------------------------------------------------
# Create output directory
# --------------------------------------------------
os.makedirs("Visualizations", exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("Anime_Data/tv_anime_ratings.csv")
df = df.dropna(subset=["year"])
df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]

# ==================================================
# COLOR PALETTES
# ==================================================
# Demographics (fixed)
demo_colors = {
    "Shoujo": "#ffb3c6",
    "Shounen": "#00b4d8",
    "Josei": "#da2c43",
    "Seinen": "#03045e"
}

# Genres (manual, editable)
genre_colors = {
    "Action": "#d00000",
    "Adventure": "#8fe388",
    "Comedy": "#ffba08",
    "Drama": "#023e8a",
    "Fantasy": "#cbff8c",
    "Romance": "#ff7b9c",
    "Sci-Fi": "#3185fc",
    "Slice of Life": "#ff9b85",
    "Mystery": "#46237a",
    "Supernatural": "#5d2e8c",
    "Other": "#1b998b"
}

# ==================================================
# DEMOGRAPHIC ANALYSIS OVER TIME
# ==================================================
df["demographics"] = df["demographics"].fillna("Unknown")
df_demo = df.assign(demographic=df["demographics"].str.split(", ")).explode("demographic")

valid_demographics = ["Shoujo", "Shounen", "Josei", "Seinen"]
df_demo = df_demo[df_demo["demographic"].isin(valid_demographics)]

demo_counts = df_demo.groupby(["year", "demographic"]).size().unstack(fill_value=0).sort_index()
demo_props = demo_counts.div(demo_counts.sum(axis=1), axis=0)
demo_order = ["Shoujo", "Shounen", "Josei", "Seinen"]

# ---------------- Plot 1: Demographics stacked density ----------------
plt.figure(figsize=(12, 7))
plt.stackplot(
    demo_props.index,
    [demo_props[d] for d in demo_order],
    labels=demo_order,
    colors=[demo_colors[d] for d in demo_order],
    alpha=0.9
)
plt.xlabel("Year")
plt.ylabel("Proportion of TV Anime")
plt.title("TV Anime Demographics Over Time (1990–2025)")
plt.legend(title="Demographic", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig("Visualizations/demographics_stacked_density.png", dpi=300)
plt.close()

# ---------------- Plot 2: Demographics 100% stacked bar with absolute counts ----------------
plt.figure(figsize=(14, 7))
bottom = None
for demo in demo_order:
    plt.bar(demo_props.index, demo_props[demo], bottom=bottom, color=demo_colors[demo], label=demo)
    bottom = demo_props[demo] if bottom is None else bottom + demo_props[demo]

for year in demo_counts.index:
    cumulative = 0
    for demo in demo_order:
        value = demo_counts.loc[year, demo]
        prop = demo_props.loc[year, demo]
        if value > 0:
            plt.text(year, cumulative + prop/2, str(value), ha="center", va="center", fontsize=7, color="white")
        cumulative += prop

plt.xlabel("Year")
plt.ylabel("Proportion of TV Anime (100%)")
plt.title("TV Anime Demographics Over Time (1990–2025)")
plt.legend(title="Demographic", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig("Visualizations/demographics_stacked_bar_absolute.png", dpi=300)
plt.close()

# ==================================================
# GENRE ANALYSIS OVER TIME
# ==================================================
df_genre = df.copy()
df_genre["genres"] = df_genre["genres"].fillna("")
df_genre = df_genre.assign(genre=df_genre["genres"].str.split(", ")).explode("genre")
df_genre = df_genre[df_genre["genre"] != ""]

TOP_N = 10
top_genres = df_genre["genre"].value_counts().head(TOP_N).index.tolist()
df_genre["genre_grouped"] = df_genre["genre"].where(df_genre["genre"].isin(top_genres), "Other")

genre_counts = df_genre.groupby(["year", "genre_grouped"]).size().unstack(fill_value=0).sort_index()
genre_props = genre_counts.div(genre_counts.sum(axis=1), axis=0)
genre_order = top_genres + ["Other"]
genre_counts = genre_counts[genre_order]
genre_props = genre_props[genre_order]

# ---------------- Plot 3: Genre stacked density ----------------
plt.figure(figsize=(14,7))
plt.stackplot(
    genre_props.index,
    [genre_props[g] for g in genre_order],
    labels=genre_order,
    colors=[genre_colors[g] for g in genre_order],
    alpha=0.9
)
plt.xlabel("Year")
plt.ylabel("Proportion of TV Anime")
plt.title(f"TV Anime Genres Over Time (1990–2025)\nTop {TOP_N} Genres + Other")
plt.legend(title="Genre", loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False)
plt.tight_layout(rect=[0,0,0.82,1])
plt.savefig("Visualizations/genres_stacked_density.png", dpi=300)
plt.close()

# ---------------- Plot 4: Genre 100% stacked bar with absolute counts ----------------
plt.figure(figsize=(14,7))
bottom = None
for g in genre_order:
    plt.bar(genre_props.index, genre_props[g], bottom=bottom, color=genre_colors[g], label=g)
    bottom = genre_props[g] if bottom is None else bottom + genre_props[g]

for year in genre_counts.index:
    cumulative = 0
    for g in genre_order:
        value = genre_counts.loc[year, g]
        prop = genre_props.loc[year, g]
        if value >= 5:
            plt.text(year, cumulative + prop/2, str(value), ha="center", va="center", fontsize=7, color="black")
        cumulative += prop

plt.xlabel("Year")
plt.ylabel("Proportion of TV Anime (100%)")
plt.title(f"TV Anime Genres Over Time (1990–2025)\nTop {TOP_N} Genres + Other")
plt.legend(title="Genre", loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False)
plt.tight_layout(rect=[0,0,0.82,1])
plt.savefig("Visualizations/genres_stacked_bar_absolute.png", dpi=300)
plt.close()

# ==================================================
# GENRE × DEMOGRAPHIC
# ==================================================
df_gd = df.copy()
df_gd["genres"] = df_gd["genres"].fillna("")
df_gd["demographics"] = df_gd["demographics"].fillna("")
df_gd = df_gd.assign(
    genre=df_gd["genres"].str.split(", "),
    demographic=df_gd["demographics"].str.split(", ")
).explode("genre").explode("demographic")

df_gd = df_gd[(df_gd["genre"] != "") & (df_gd["demographic"].isin(valid_demographics)) & (df_gd["genre"].isin(top_genres))]

gd_counts = df_gd.groupby(["demographic","genre"]).size().unstack(fill_value=0).reindex(index=demo_order, columns=top_genres, fill_value=0)
gd_props = gd_counts.div(gd_counts.sum(axis=1), axis=0)
x = np.arange(len(gd_counts))

# ---------------- Plot 5: Genre x Demographic 100% stacked bar with absolute counts ----------------
plt.figure(figsize=(12,7))
bottom = np.zeros(len(gd_counts))
for g in top_genres:
    plt.bar(x, gd_props[g], bottom=bottom, color=genre_colors[g], label=g)
    bottom += gd_props[g].values

for i, demo in enumerate(demo_order):
    cumulative = 0
    for g in top_genres:
        value = gd_counts.loc[demo,g]
        prop = gd_props.loc[demo,g]
        if value > 0:
            plt.text(i, cumulative + prop/2, str(value), ha="center", va="center", fontsize=8, color="white")
        cumulative += prop

plt.xticks(x, demo_order)
plt.xlabel("Demographic")
plt.ylabel("Proportion of TV Anime (100%)")
plt.title(f"Genre Distribution by Demographic (1990–2025)\nTop {TOP_N} Genres")
plt.legend(title="Genre", loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False)
plt.tight_layout(rect=[0,0,0.82,1])
plt.savefig("Visualizations/genre_by_demographic_100pct_absolute.png", dpi=300)
plt.close()

# ==================================================
# GENRE × DEMOGRAPHIC × YEAR
# ==================================================
# ---------------- Plot 6: Genres over time by demographic (stacked density plot) ----------------
df_gtd = df.copy()
df_gtd["genres"] = df_gtd["genres"].fillna("")
df_gtd["demographics"] = df_gtd["demographics"].fillna("")
df_gtd = df_gtd.assign(
    genre=df_gtd["genres"].str.split(", "),
    demographic=df_gtd["demographics"].str.split(", ")
).explode("genre").explode("demographic")

# Keep only valid demographics and top genres
df_gtd = df_gtd[
    (df_gtd["demographic"].isin(valid_demographics)) &
    (df_gtd["genre"].isin(top_genres))
]

# Aggregate counts per year
df_gtd_counts = df_gtd.groupby(["year", "demographic", "genre"]).size().reset_index(name="count")

# Pivot for stacking
fig, axes = plt.subplots(1, len(valid_demographics), figsize=(20,6), sharey=True)

for i, demo in enumerate(valid_demographics):
    df_demo_genre = df_gtd_counts[df_gtd_counts["demographic"] == demo].pivot(
        index="year", columns="genre", values="count"
    ).fillna(0)

    # Convert to proportions (100% stacked)
    df_demo_genre_props = df_demo_genre.div(df_demo_genre.sum(axis=1), axis=0)

    x = df_demo_genre_props.index
    y_values = [df_demo_genre_props[g].values for g in top_genres]

    axes[i].stackplot(
        x,
        y_values,
        labels=top_genres,
        colors=[genre_colors[g] for g in top_genres],
        alpha=0.9
    )

    axes[i].set_title(demo)
    axes[i].set_xlabel("Year")
    if i == 0:
        axes[i].set_ylabel("Proportion of TV Anime (100%)")

# Shared legend
axes[-1].legend(title="Genre", bbox_to_anchor=(1.05,1), frameon=False)
plt.suptitle(f"Top Genres Over Time by Demographic (1990–2025)")
plt.tight_layout(rect=[0,0,0.95,0.95])
plt.savefig("Visualizations/genres_over_time_by_demographic_100pct.png", dpi=300)
plt.close()


# ---------------- Plot 7: Genres over time by demographic (100% stacked bars with absolute counts) ----------------
fig, axes = plt.subplots(1, len(valid_demographics), figsize=(20, 6), sharey=True)

for i, demo in enumerate(valid_demographics):
    df_demo_genre = df_gtd_counts[df_gtd_counts["demographic"] == demo].pivot(
        index="year", columns="genre", values="count"
    ).fillna(0)

    # Convert to proportions (100% stacked)
    df_demo_genre_props = df_demo_genre.div(df_demo_genre.sum(axis=1), axis=0)

    x = np.arange(len(df_demo_genre_props.index))
    bottom = np.zeros(len(df_demo_genre_props))

    for g in top_genres:
        axes[i].bar(
            x,
            df_demo_genre_props[g],
            bottom=bottom,
            color=genre_colors[g],
            label=g
        )
        bottom += df_demo_genre_props[g].values

    # Annotate absolute counts
    for xi, year in enumerate(df_demo_genre.index):
        cumulative = 0
        for g in top_genres:
            abs_count = df_demo_genre.loc[year, g]
            prop = df_demo_genre_props.loc[year, g]
            if abs_count > 0:
                axes[i].text(
                    xi,
                    cumulative + prop / 2,
                    str(int(abs_count)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white"
                )
            cumulative += prop

    axes[i].set_title(demo)
    axes[i].set_xlabel("Year")
    if i == 0:
        axes[i].set_ylabel("Proportion of TV Anime (100%)")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(df_demo_genre.index, rotation=45, ha="right")

# Shared legend
axes[-1].legend(title="Genre", bbox_to_anchor=(1.05, 1), frameon=False)
plt.suptitle(f"Top Genres Over Time by Demographic (1990–2025) — 100% Stacked Bars")
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig("Visualizations/genres_over_time_by_demographic_100pct_bars.png", dpi=300)
plt.close()



# ==================================================
# Regression analysis: predict year from genre and demographic
# ==================================================

# Prepare the data
df_reg = df_gd.copy()  # Use the exploded genre × demographic dataset
df_reg = df_reg[(df_reg["genre"] != "") & (df_reg["demographic"].isin(valid_demographics))]

# Convert year to numeric (should already be numeric, but just to be safe)
df_reg["year"] = pd.to_numeric(df_reg["year"])

# Define formula: year ~ genre + demographic + genre:demographic
# This will include main effects and interaction
formula = "year ~ C(genre) + C(demographic)"

# Fit linear regression
model = smf.ols(formula=formula, data=df_reg).fit()

# Print summary to console
print("\n================ Regression Analysis ================\n")
print(model.summary())
print("\n====================================================\n")

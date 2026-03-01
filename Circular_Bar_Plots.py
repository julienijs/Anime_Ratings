import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# --------------------------------------------------
# Setup
# --------------------------------------------------
output_dir = "Visualizations/Circular_By_Year"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("Anime_Data/tv_anime_ratings.csv")

df = df.dropna(subset=["score", "demographics", "year"])
df = df[(df["year"] >= 1990) & (df["year"] <= 2025)]

# 🔹 FIX 1: Force year to integer
df["year"] = df["year"].astype(int)

# Explode genres
df["genres"] = df["genres"].fillna("")
df = df.assign(genre=df["genres"].str.split(", ")).explode("genre")
df = df[df["genre"] != ""]

# Explode demographics
df = df.assign(demographic=df["demographics"].str.split(", ")).explode("demographic")

demo_order = ["Shounen", "Shoujo", "Seinen", "Josei"]
df = df[df["demographic"].isin(demo_order)]

# --------------------------------------------------
# Stable genre color mapping
# --------------------------------------------------
all_genres = sorted(df["genre"].unique())
cmap = mpl.colormaps["tab20"]
colors = cmap(np.linspace(0, 1, len(all_genres)))
genre_colors = dict(zip(all_genres, colors))

# --------------------------------------------------
# Layout parameters
# --------------------------------------------------
GROUP_GAP = 3
INNER_RADIUS = 3
BAR_SCALING = 0.5

years = sorted(df["year"].unique())

# --------------------------------------------------
# Loop over years
# --------------------------------------------------
for year in years:

    year_df = df[df["year"] == year]

    agg = (
        year_df.groupby(["demographic", "genre"])["score"]
               .mean()
               .reset_index()
    )

    if agg.empty:
        continue

    agg["demographic"] = pd.Categorical(
        agg["demographic"],
        categories=demo_order,
        ordered=True
    )

    agg = agg.sort_values(["demographic", "genre"])

    values, labels, groups = [], [], []

    for demo in demo_order:
        sub = agg[agg["demographic"] == demo]

        for _, row in sub.iterrows():
            values.append(row["score"])
            labels.append(row["genre"])
            groups.append(demo)

        for _ in range(GROUP_GAP):
            values.append(0)
            labels.append("")
            groups.append("")

    if not values:
        continue

    N = len(values)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    values = np.array(values)
    values_scaled = values * BAR_SCALING

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))

    # --------------------------------------------------
    # Draw bars
    # --------------------------------------------------
    bars = ax.bar(
        angles,
        values_scaled,
        width=2*np.pi/N * 0.92,
        bottom=INNER_RADIUS,
        edgecolor="white",
        linewidth=0.8
    )

    for i, (bar, label) in enumerate(zip(bars, labels)):
        if label == "":
            bar.set_alpha(0)
        else:
            bar.set_color(genre_colors[label])

    # --------------------------------------------------
    # Mean score inside bars
    # --------------------------------------------------
    for angle, value, label in zip(angles, values_scaled, labels):
        if label == "" or value == 0:
            continue

        ax.text(
            angle,
            INNER_RADIUS + value/2,
            f"{value/BAR_SCALING:.1f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold"
        )

    # --------------------------------------------------
    # FIX 2: Make demographics visually explicit
    # --------------------------------------------------
    start = 0
    max_radius = INNER_RADIUS + max(values_scaled) + 0.8

    for demo in demo_order:
        group_size = len(agg[agg["demographic"] == demo])
        if group_size == 0:
            continue

        end = start + group_size

        # Thick separator line
        ax.plot(
            [angles[start], angles[start]],
            [INNER_RADIUS, max_radius],
            color="black",
            linewidth=2
        )

        # Subtle sector background arc
        theta = np.linspace(angles[start], angles[end-1], 200)
        ax.fill_between(
            theta,
            INNER_RADIUS * 0.2,
            INNER_RADIUS,
            alpha=0.05
        )

        # Large demographic label
        mid_angle = np.mean(angles[start:end])
        ax.text(
            mid_angle,
            INNER_RADIUS * 0.55,
            demo.upper(),
            fontsize=14,
            fontweight=600,
            alpha=0.9,
            ha="center",
            va="center"
        )

        start = end + GROUP_GAP

    # --------------------------------------------------
    # Year in center
    # --------------------------------------------------
    ax.text(
        0,
        0,
        str(year),
        fontsize=28,
        fontweight="bold",
        ha="center",
        va="center"
    )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial reference lines
    for r in np.linspace(INNER_RADIUS, INNER_RADIUS + max(values_scaled), 4):
        ax.plot(np.linspace(0, 2*np.pi, 300),
                [r]*300,
                color="lightgrey",
                linewidth=0.5,
                alpha=0.3)

    # Legend
    legend_elements = [
        mpl.patches.Patch(facecolor=genre_colors[g], label=g)
        for g in all_genres
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.15, 0.5),
        loc="center left",
        title="Genre",
        frameon=False,
        fontsize=8
    )

    plt.title("Mean Score by Genre and Demographic", pad=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/circular_{year}.png", dpi=300)
    plt.close()

print("Final circular plots generated.")
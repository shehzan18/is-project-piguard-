import os
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for WSL
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# CREATE OUTPUT FOLDER
# =========================
OUTPUT_DIR = "results_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# OUR RESULTS
# =========================
our_results = pd.DataFrame({
    "Model": ["WITH MOF", "WITHOUT MOF", "PIGuard++"],
    "NotInject": [84.96, 74.63, 81.68],
    "WildGuard": [72.89, 83.01, 90.32],
    "BIPIA": [50.00, 50.67, 51.33]
})

# =========================
# PAPER RESULTS
# =========================
paper_results = pd.DataFrame({
    "Metric": ["NotInject", "WildGuard", "BIPIA"],
    "Paper PIGuard": [89.38, 76.11, 68.34],
    "Our Reproduction": [84.96, 72.89, 50.00]
})

# =========================
# MOF COMPARISON
# =========================
mof_results = pd.DataFrame({
    "Metric": ["NotInject", "WildGuard", "BIPIA"],
    "WITHOUT MOF": [74.63, 83.01, 50.67],
    "WITH MOF": [84.96, 72.89, 50.00]
})

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (9, 5)
})

# =========================
# CHART 1: OUR MODELS
# =========================
metrics = ["NotInject", "WildGuard", "BIPIA"]

for metric in metrics:
    plt.figure()

    plt.bar(
        our_results["Model"],
        our_results[metric]
    )

    plt.title(f"{metric} Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Models")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    for i, v in enumerate(our_results[metric]):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{metric.lower()}_comparison.png", dpi=300)
    plt.close()

# =========================
# CHART 2: PAPER VS OURS
# =========================
plt.figure()

x = range(len(paper_results["Metric"]))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    paper_results["Paper PIGuard"],
    width=width,
    label="Paper PIGuard"
)

plt.bar(
    [i + width/2 for i in x],
    paper_results["Our Reproduction"],
    width=width,
    label="Our Reproduction"
)

plt.xticks(x, paper_results["Metric"])
plt.ylabel("Accuracy (%)")
plt.xlabel("Metrics")
plt.title("Paper vs Our Reproduction")
plt.ylim(0, 100)
plt.legend()

for i, v in enumerate(paper_results["Paper PIGuard"]):
    plt.text(i - width/2, v + 1, f"{v:.2f}%", ha="center")

for i, v in enumerate(paper_results["Our Reproduction"]):
    plt.text(i + width/2, v + 1, f"{v:.2f}%", ha="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/paper_vs_ours.png", dpi=300)
plt.close()

# =========================
# CHART 3: MOF IMPACT
# =========================
plt.figure()

x = range(len(mof_results["Metric"]))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    mof_results["WITHOUT MOF"],
    width=width,
    label="WITHOUT MOF"
)

plt.bar(
    [i + width/2 for i in x],
    mof_results["WITH MOF"],
    width=width,
    label="WITH MOF"
)

plt.xticks(x, mof_results["Metric"])
plt.ylabel("Accuracy (%)")
plt.xlabel("Metrics")
plt.title("Impact of MOF")
plt.ylim(0, 100)
plt.legend()

for i, v in enumerate(mof_results["WITHOUT MOF"]):
    plt.text(i - width/2, v + 1, f"{v:.2f}%", ha="center")

for i, v in enumerate(mof_results["WITH MOF"]):
    plt.text(i + width/2, v + 1, f"{v:.2f}%", ha="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mof_impact.png", dpi=300)
plt.close()

print(f"Charts saved in: {OUTPUT_DIR}")
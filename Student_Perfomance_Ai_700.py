# ============================================================
#         STUDENT PERFORMANCE PREDICTOR
#         Uses Machine Learning (Decision Tree) to predict
#         a student's academic level based on their habits.
# ============================================================

import pandas as pd
import json
import os
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from colorama import Fore, Style, init
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Initialize colorama so colors work on Windows too
init(autoreset=True)

# ─────────────────────────────────────────────────────────────
#  SECTION 1 — TRAINING DATASET
#  This is the data the model learns from.
#  Each row is one student with their study stats and outcome.
# ─────────────────────────────────────────────────────────────
TRAINING_DATA = {
    "StudyHours":    [2, 4, 6, 1, 8, 5, 3, 7, 2, 9, 4, 6, 1, 8, 5],
    "Attendance":    [60,75,85,50,95,80,65,90,55,98,70,88,45,92,78],
    "PreviousGrade": [50,65,75,40,85,70,60,80,45,90,68,82,35,88,73],
    "Assignments":   [55,70,80,45,90,75,65,85,50,95,72,88,40,92,78],
    "Level": [
        "Average","Good","Good","At Risk","Excellent",
        "Good","Average","Excellent","At Risk","Excellent",
        "Good","Good","At Risk","Excellent","Good"
    ]
}

# ─────────────────────────────────────────────────────────────
#  SECTION 2 — LEVEL MAPPING
#  We convert text labels → numbers for the ML model.
#  The model only understands numbers, not words.
# ─────────────────────────────────────────────────────────────
LEVEL_MAP     = {"At Risk": 0, "Average": 1, "Good": 2, "Excellent": 3}
LEVEL_MAP_REV = {0: "At Risk", 1: "Average", 2: "Good", 3: "Excellent"}

LEVEL_COLORS = {
    "At Risk":   Fore.RED,
    "Average":   Fore.YELLOW,
    "Good":      Fore.CYAN,
    "Excellent": Fore.GREEN,
}

LEVEL_EMOJIS = {
    "At Risk":   "🔴",
    "Average":   "🟡",
    "Good":      "🔵",
    "Excellent": "🟢",
}

# ─────────────────────────────────────────────────────────────
#  HISTORY FILE PATH
#  We save to the Desktop to avoid OneDrive permission errors.
#  os.path.expanduser("~") gives your Windows user folder, e.g.
#  C:\Users\ECON — then we add \Desktop\student_history.json
# ─────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(
    os.path.expanduser("~"), "Desktop", "student_history.json"
)

# ─────────────────────────────────────────────────────────────
#  SECTION 3 — MODEL TRAINING
#  We prepare the data and train the Decision Tree model.
# ─────────────────────────────────────────────────────────────
def train_model():
    """
    Prepares the dataset, trains a Decision Tree classifier,
    and returns the trained model along with accuracy info.
    """
    df = pd.DataFrame(TRAINING_DATA)
    df["Level"] = df["Level"].map(LEVEL_MAP)

    features = ["StudyHours", "Attendance", "PreviousGrade", "Assignments"]
    X = df[features]
    y = df["Level"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Cross-validation: tests the model on different data splits
    # to see how reliable it is (higher = better)
    cv_scores = cross_val_score(model, X, y, cv=3)
    accuracy = round(cv_scores.mean() * 100, 1)

    # Feature importance: which factor affects the prediction most?
    importances = dict(zip(features, model.feature_importances_))

    return model, accuracy, importances


# ─────────────────────────────────────────────────────────────
#  SECTION 4 — INPUT HELPERS
#  Safe input functions that keep asking until valid data entered.
# ─────────────────────────────────────────────────────────────
def get_number(prompt, min_val, max_val):
    """
    Asks the user for a number within [min_val, max_val].
    Keeps looping until a valid number is entered.
    """
    while True:
        try:
            value = float(input(Fore.WHITE + prompt))
            if min_val <= value <= max_val:
                return value
            print(Fore.RED + f"  ⚠  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print(Fore.RED + "  ⚠  That's not a number. Try again.")


def get_student_name():
    """Asks for a student name (cannot be empty)."""
    while True:
        name = input(Fore.WHITE + "  Enter Student Name: ").strip()
        if name:
            return name
        print(Fore.RED + "  ⚠  Name cannot be empty.")


# ─────────────────────────────────────────────────────────────
#  SECTION 5 — PREDICTION
#  Uses the trained model to predict the student's level.
# ─────────────────────────────────────────────────────────────
def predict(model, study, attendance, grade, assignments):
    """
    Feeds the student's data into the ML model and returns:
    - level name (e.g. "Good")
    - level index (0–3)
    - probability scores for each level (confidence %)
    """
    input_data = [[study, attendance, grade, assignments]]
    level_index = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]  # confidence per class

    level_name = LEVEL_MAP_REV[level_index]
    return level_name, level_index, probabilities


# ─────────────────────────────────────────────────────────────
#  SECTION 6 — HEALTH & STUDY ADJUSTMENT
#  Adjusts the predicted level based on sleep & study habits.
#  Also returns personalized health and study tips.
# ─────────────────────────────────────────────────────────────
def apply_health_adjustment(level_index, study_hours, sleep_hours):
    """
    Checks for unhealthy patterns and:
    - Lowers the level prediction if health is poor
    - Returns a list of personalized tips
    """
    tips = []
    penalty = 0

    # ── Sleep check ──────────────────────────────────────────
    if sleep_hours < 5:
        tips.append(("🛌 Sleep", "Very low sleep! Aim for 7–8 hrs. Poor sleep hurts memory.", "danger"))
        penalty += 1
    elif sleep_hours < 7:
        tips.append(("🛌 Sleep", "Sleep is below recommended. Try to get at least 7 hours.", "warning"))
    else:
        tips.append(("🛌 Sleep", "Great sleep schedule! This helps with memory and focus.", "good"))

    # ── Study hours check ─────────────────────────────────────
    if study_hours > 10:
        tips.append(("📚 Study", "You're over-studying! Fatigue reduces effectiveness. Take breaks.", "danger"))
        penalty += 1
    elif study_hours < 2:
        tips.append(("📚 Study", "Very low study time. Try to study at least 2–3 hours daily.", "warning"))
    elif 4 <= study_hours <= 8:
        tips.append(("📚 Study", "Ideal study hours. Keep it consistent!", "good"))
    else:
        tips.append(("📚 Study", "Decent study time. Maintain a regular schedule.", "neutral"))

    # ── Study/sleep balance ───────────────────────────────────
    free_time = 24 - study_hours - sleep_hours
    if free_time < 4:
        tips.append(("⚖ Balance", "Very little free time. Reserve time for food, exercise, breaks.", "warning"))
    else:
        tips.append(("⚖ Balance", f"You have ~{free_time:.0f} hrs free time. Use some for exercise!", "neutral"))

    adjusted_index = max(0, level_index - penalty)
    return LEVEL_MAP_REV[adjusted_index], adjusted_index, tips


# ─────────────────────────────────────────────────────────────
#  SECTION 7 — GRADE-BASED RECOMMENDATIONS
#  Personalized advice based on the final predicted level.
# ─────────────────────────────────────────────────────────────
LEVEL_ADVICE = {
    "At Risk": [
        "📌 Attend all classes — attendance is your first priority.",
        "📌 Complete every assignment, even if imperfect.",
        "📌 Speak to your teacher or tutor for extra help.",
        "📌 Break your study into 25-min sessions (Pomodoro technique).",
        "📌 Remove distractions: phone-free study zone.",
    ],
    "Average": [
        "📌 Increase daily study by 30–45 minutes.",
        "📌 Review your notes within 24 hours of each class.",
        "📌 Focus on your weakest subject first each session.",
        "📌 Join a study group for accountability.",
        "📌 Practice past papers to improve exam confidence.",
    ],
    "Good": [
        "📌 Challenge yourself with harder problems.",
        "📌 Teach concepts to others — it deepens your understanding.",
        "📌 Set a target to reach 'Excellent' next term.",
        "📌 Start assignments early to leave time for revision.",
        "📌 Track your weekly progress in a journal.",
    ],
    "Excellent": [
        "📌 Mentor struggling classmates — great for leadership skills.",
        "📌 Explore advanced topics or online courses in your subject.",
        "📌 Maintain your routine — consistency is your superpower.",
        "📌 Set new goals: competitions, scholarships, or projects.",
        "📌 Don't burn out — keep balance and enjoy the journey!",
    ],
}


# ─────────────────────────────────────────────────────────────
#  SECTION 8 — HISTORY (SAVE & LOAD)
#  Saves each prediction to a JSON file so you can track
#  multiple students or revisit old results.
# ─────────────────────────────────────────────────────────────
def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
    except (PermissionError, json.JSONDecodeError) as e:
        print(Fore.RED + f"  ⚠  Could not load history: {e}")
    return []


def save_to_history(record):
    try:
        history = load_history()
        history.append(record)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except PermissionError:
        print(Fore.RED + "  ⚠  Permission denied — could not save history.")
        print(Fore.YELLOW + "     Move your project OUT of OneDrive to fix this.")


def view_history():
    """Displays all past predictions in a neat table."""
    history = load_history()
    if not history:
        print(Fore.YELLOW + "\n  No history yet. Make a prediction first!\n")
        return

    print(Fore.CYAN + "\n" + "─"*60)
    print(Fore.CYAN + "  📋  PREDICTION HISTORY")
    print(Fore.CYAN + "─"*60)

    rows = []
    for i, r in enumerate(history, 1):
        rows.append([
            i,
            r.get("name", "N/A"),
            r.get("date", "N/A"),
            r.get("ml_level", "N/A"),
            r.get("final_level", "N/A"),
            f"{r.get('study_hours', '?')} hrs",
            f"{r.get('attendance', '?')}%",
        ])

    headers = ["#", "Name", "Date", "ML Level", "Final Level", "Study", "Attend."]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print()


def clear_history():
    """Deletes all saved predictions after confirmation."""
    confirm = input(Fore.RED + "  ⚠  Delete ALL history? (yes/no): ").strip().lower()
    if confirm == "yes":
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        print(Fore.GREEN + "  ✔  History cleared.\n")
    else:
        print(Fore.YELLOW + "  Cancelled.\n")


# ─────────────────────────────────────────────────────────────
#  SECTION 9 — DISPLAY HELPERS
#  Functions that print formatted, colorful output.
# ─────────────────────────────────────────────────────────────
def print_banner():
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════╗
║       🎓  STUDENT PERFORMANCE PREDICTOR  🎓             ║
║          Powered by Machine Learning (Decision Tree)     ║
╚══════════════════════════════════════════════════════════╝""")


def print_section(title):
    print(Fore.CYAN + f"\n{'─'*60}")
    print(Fore.CYAN + f"  {title}")
    print(Fore.CYAN + "─"*60)


def print_confidence_bar(probabilities, classes):
    """Shows a visual bar for each predicted level's confidence."""
    print_section("📊 MODEL CONFIDENCE SCORES")
    for i, cls in enumerate(classes):
        pct = probabilities[i] * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = LEVEL_COLORS.get(LEVEL_MAP_REV.get(cls, ""), Fore.WHITE)
        label = LEVEL_MAP_REV.get(cls, str(cls))
        print(f"  {color}{label:<12} {bar} {pct:5.1f}%")


def print_health_tips(tips):
    print_section("💡 HEALTH & BALANCE TIPS")
    for category, message, status in tips:
        if status == "good":
            color = Fore.GREEN
            icon = "✔"
        elif status == "warning":
            color = Fore.YELLOW
            icon = "⚠"
        elif status == "danger":
            color = Fore.RED
            icon = "✘"
        else:
            color = Fore.WHITE
            icon = "•"
        print(f"  {color}{icon} {category}: {message}")


def print_recommendations(level):
    print_section(f"📚 RECOMMENDATIONS  [{level}]")
    for tip in LEVEL_ADVICE[level]:
        print(f"  {Fore.WHITE}{tip}")


def print_feature_importance(importances):
    print_section("🔍 WHAT MATTERS MOST (Feature Importance)")
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    labels = {
        "StudyHours": "Study Hours",
        "Attendance": "Attendance",
        "PreviousGrade": "Previous Grade",
        "Assignments": "Assignment Score",
    }
    for feature, score in sorted_items:
        pct = score * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {Fore.MAGENTA}{labels.get(feature, feature):<20} {bar} {pct:5.1f}%")
    print(f"\n  {Fore.WHITE}This shows which factor the model relies on most.")


def print_summary_card(name, inputs, ml_level, final_level, accuracy):
    """Prints a clean result card for the student."""
    print_section(f"🎯 RESULT CARD — {name.upper()}")
    emoji = LEVEL_EMOJIS.get(final_level, "")
    color = LEVEL_COLORS.get(final_level, Fore.WHITE)

    rows = [
        ["Study Hours/day",   f"{inputs['study_hours']} hrs"],
        ["Sleep Hours/day",   f"{inputs['sleep_hours']} hrs"],
        ["Attendance",        f"{inputs['attendance']}%"],
        ["Previous Grade",    f"{inputs['previous_grade']}%"],
        ["Assignment Score",  f"{inputs['assignments']}%"],
        ["─────────────────", "──────────────"],
        ["ML Prediction",     ml_level],
        ["After Health Adj.", f"{emoji} {final_level}"],
        ["Model Accuracy",    f"~{accuracy}%"],
    ]
    print(tabulate(rows, tablefmt="rounded_outline"))
    print(f"\n  {color}{Style.BRIGHT}  ➤  Final Performance Level: {emoji}  {final_level}  {emoji}\n")


# ─────────────────────────────────────────────────────────────
#  SECTION 10 — GRAPHS
#  Generates a 4-panel matplotlib figure showing:
#    1. Student scores vs level benchmarks (bar chart)
#    2. Model confidence for each level (horizontal bar)
#    3. Daily time breakdown: study / sleep / free (pie chart)
#    4. Feature importance (horizontal bar chart)
#  The chart is saved to Desktop and also shown on screen.
# ─────────────────────────────────────────────────────────────

# Colors used in all graphs (matches the dark terminal theme)
GRAPH_COLORS = {
    "At Risk":   "#e74c3c",   # red
    "Average":   "#f39c12",   # orange
    "Good":      "#3498db",   # blue
    "Excellent": "#2ecc71",   # green
    "study":     "#9b59b6",   # purple
    "sleep":     "#1abc9c",   # teal
    "free":      "#95a5a6",   # grey
}

# Benchmark scores for each performance level
# (used as reference lines on the student score chart)
BENCHMARKS = {
    "At Risk":   {"Attendance": 52, "Prev. Grade": 42, "Assignments": 47},
    "Average":   {"Attendance": 62, "Prev. Grade": 55, "Assignments": 60},
    "Good":      {"Attendance": 80, "Prev. Grade": 73, "Assignments": 77},
    "Excellent": {"Attendance": 93, "Prev. Grade": 88, "Assignments": 91},
}


def generate_graphs(name, inputs, final_level, probabilities, model_classes, importances):
    """
    Builds a 2×2 grid of charts and displays + saves them.

    Parameters:
        name         — student's name (used in title)
        inputs       — dict with study_hours, sleep_hours, attendance, etc.
        final_level  — final predicted level string (e.g. "Good")
        probabilities— numpy array of confidence scores from model
        model_classes— class indices from model.classes_
        importances  — dict of feature → importance score
    """

    # ── Dark theme setup ──────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")           # deep navy background
    fig.suptitle(
        f"Performance Report  —  {name}   |   Level: {final_level}",
        fontsize=14, fontweight="bold", color="white", y=0.98
    )

    level_color = GRAPH_COLORS.get(final_level, "#ffffff")

    # ── Shared axes style helper ──────────────────────────────
    def style_ax(ax, title):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ══════════════════════════════════════════════════════════
    #  CHART 1 — Student Scores vs Level Benchmarks
    #  Shows attendance, previous grade, and assignment score
    #  as bars, with a dashed line showing the target benchmark.
    # ══════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(2, 2, 1)
    style_ax(ax1, "Your Scores vs Benchmarks")

    categories   = ["Attendance", "Prev. Grade", "Assignments"]
    student_vals = [inputs["attendance"], inputs["previous_grade"], inputs["assignments"]]
    bench_vals   = [BENCHMARKS[final_level][c] for c in categories]
    x            = np.arange(len(categories))
    bar_w        = 0.35

    bars1 = ax1.bar(x - bar_w/2, student_vals, bar_w, label="Your Score",
                    color=level_color, alpha=0.9, zorder=3)
    bars2 = ax1.bar(x + bar_w/2, bench_vals,   bar_w, label=f"{final_level} Benchmark",
                    color="#555577", alpha=0.8, zorder=3)

    # Value labels on top of each bar
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", va="bottom",
                 color="white", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", va="bottom",
                 color="#aaaacc", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, color="white")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Score (%)", color="white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax1.yaxis.grid(True, color="#333355", linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)

    # ══════════════════════════════════════════════════════════
    #  CHART 2 — Model Confidence (how sure was the ML model?)
    #  Horizontal bars showing probability % for each level.
    # ══════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(2, 2, 2)
    style_ax(ax2, "Model Confidence per Level")

    level_names = [LEVEL_MAP_REV[c] for c in model_classes]
    bar_colors  = [GRAPH_COLORS.get(lvl, "#888888") for lvl in level_names]
    pct_vals    = [p * 100 for p in probabilities]
    y_pos       = np.arange(len(level_names))

    h_bars = ax2.barh(y_pos, pct_vals, color=bar_colors, alpha=0.9, zorder=3)

    # Value labels at end of each bar
    for i, (bar, val) in enumerate(zip(h_bars, pct_vals)):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", color="white", fontsize=10, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(level_names, color="white", fontsize=10)
    ax2.set_xlim(0, 120)
    ax2.set_xlabel("Confidence (%)", color="white")
    ax2.xaxis.grid(True, color="#333355", linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)

    # Highlight the predicted level's bar with a border
    for i, lvl in enumerate(level_names):
        if lvl == final_level:
            h_bars[i].set_edgecolor("white")
            h_bars[i].set_linewidth(2)

    # ══════════════════════════════════════════════════════════
    #  CHART 3 — Daily Time Breakdown (Pie Chart)
    #  Shows how the student splits 24 hours: study / sleep / free
    # ══════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(2, 2, 3)
    style_ax(ax3, "Daily Time Breakdown (24 hrs)")

    study = inputs["study_hours"]
    sleep = inputs["sleep_hours"]
    free  = max(0, 24 - study - sleep)

    time_labels = [f"Study\n{study:.0f} hrs", f"Sleep\n{sleep:.0f} hrs", f"Free\n{free:.0f} hrs"]
    time_vals   = [study, sleep, free]
    time_colors = [GRAPH_COLORS["study"], GRAPH_COLORS["sleep"], GRAPH_COLORS["free"]]

    # Remove zero slices so pie doesn't error
    filtered = [(v, l, c) for v, l, c in zip(time_vals, time_labels, time_colors) if v > 0]
    f_vals, f_labels, f_colors = zip(*filtered) if filtered else ([], [], [])

    wedges, texts, autotexts = ax3.pie(
        f_vals, labels=f_labels, colors=f_colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    # Sleep warning ring if sleep < 7
    if sleep < 7:
        ax3.text(0, -1.45, "⚠ Aim for 7–8 hrs sleep!", ha="center",
                 color="#f39c12", fontsize=9, fontweight="bold")

    # ══════════════════════════════════════════════════════════
    #  CHART 4 — Feature Importance
    #  Which of the 4 inputs did the ML model rely on most?
    # ══════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(2, 2, 4)
    style_ax(ax4, "What Influenced the Prediction Most?")

    feat_labels = {
        "StudyHours":    "Study Hours",
        "Attendance":    "Attendance",
        "PreviousGrade": "Previous Grade",
        "Assignments":   "Assignments",
    }
    sorted_feats = sorted(importances.items(), key=lambda x: x[1])
    f_names  = [feat_labels.get(k, k) for k, _ in sorted_feats]
    f_scores = [v * 100 for _, v in sorted_feats]

    # Gradient colors from grey (low) to level color (high)
    bar_palette = ["#555577", "#7777aa", "#9999cc", level_color]
    f_bars = ax4.barh(f_names, f_scores, color=bar_palette[:len(f_names)],
                      alpha=0.9, zorder=3)

    for bar, val in zip(f_bars, f_scores):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", color="white", fontsize=10)

    ax4.set_xlim(0, 110)
    ax4.set_xlabel("Importance (%)", color="white")
    ax4.xaxis.grid(True, color="#333355", linestyle="--", alpha=0.6)
    ax4.set_axisbelow(True)

    # ── Layout & Save ─────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])   # leave space for suptitle

    # Save chart — try Desktop first, then Downloads as fallback
    filename = f"report_{name.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.png"
    for folder in ["Desktop", "Downloads"]:
        chart_path = os.path.join(os.path.expanduser("~"), folder, filename)
        try:
            plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            print(Fore.GREEN + f"\n  ✔  Chart saved → {chart_path}")
            print(Fore.CYAN  +  "  Opening chart in your image viewer...\n")
            break
        except Exception:
            chart_path = None
            continue

    if chart_path is None:
        print(Fore.RED + "\n  ✘  Could not save chart to Desktop or Downloads.")
        return

    try:
        # ── Open the saved PNG using the OS default viewer ────
        # This is the most reliable way on Windows/Mac/Linux.
        # It opens Windows Photos, Preview, or eog — always works.
        import subprocess, platform
        system = platform.system()

        if system == "Windows":
            os.startfile(chart_path)              # Windows Photos
        elif system == "Darwin":
            subprocess.run(["open", chart_path])  # macOS Preview
        else:
            subprocess.run(["xdg-open", chart_path])  # Linux

    except Exception as e:
        print(Fore.RED + f"\n  ✘  Could not open chart: {e}")
        print(Fore.YELLOW + "     But the PNG was saved — open it manually from Desktop.")


# ─────────────────────────────────────────────────────────────
#  SECTION 11 — MAIN PROGRAM
#  This is where everything connects together.
# ─────────────────────────────────────────────────────────────
def run_prediction(model, accuracy, importances):
    """Collects student info, runs prediction, displays results."""
    print_section("📝 ENTER STUDENT DETAILS")

    name = get_student_name()

    study_hours  = get_number("  Study Hours per day   (0–24): ", 0, 24)
    sleep_hours  = get_number("  Sleep Hours per day   (0–12): ", 0, 12)

    # Prevent impossible schedule
    if study_hours + sleep_hours > 24:
        print(Fore.YELLOW + f"\n  ⚠  Study + Sleep > 24 hrs. Adjusting sleep to fit.")
        sleep_hours = max(0, 24 - study_hours)
        print(Fore.YELLOW + f"     Sleep adjusted to {sleep_hours:.1f} hrs.\n")

    attendance     = get_number("  Attendance            (0–100%): ", 0, 100)
    previous_grade = get_number("  Previous Grade        (0–100%): ", 0, 100)
    assignments    = get_number("  Assignment Score      (0–100%): ", 0, 100)

    # ── Run the ML model ─────────────────────────────────────
    ml_level, ml_index, probabilities = predict(
        model, study_hours, attendance, previous_grade, assignments
    )

    # ── Apply health adjustments ──────────────────────────────
    final_level, final_index, health_tips = apply_health_adjustment(
        ml_index, study_hours, sleep_hours
    )

    # ── Display everything ────────────────────────────────────
    inputs = {
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "attendance": attendance,
        "previous_grade": previous_grade,
        "assignments": assignments,
    }

    print_summary_card(name, inputs, ml_level, final_level, accuracy)
    print_confidence_bar(probabilities, model.classes_)
    print_health_tips(health_tips)
    print_recommendations(final_level)
    print_feature_importance(importances)

    # ── Generate graphs ───────────────────────────────────────
    generate_graphs(name, inputs, final_level, probabilities, model.classes_, importances)

    # ── Save to history ───────────────────────────────────────
    record = {
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ml_level": ml_level,
        "final_level": final_level,
        **inputs,
    }
    save_to_history(record)
    print(Fore.GREEN + f"\n  ✔  Result saved to '{HISTORY_FILE}'\n")


def main():
    """Main menu loop — keeps the program running until user exits."""
    print_banner()

    # Train model once at startup
    print(Fore.YELLOW + "\n  ⏳ Training model...", end="")
    model, accuracy, importances = train_model()
    print(Fore.GREEN + f" Done! (Model accuracy: ~{accuracy}%)\n")

    while True:
        print(Fore.CYAN + "─"*60)
        print(Fore.WHITE + "  MAIN MENU")
        print(Fore.CYAN + "─"*60)
        print("  [1]  🎓 Predict Student Performance")
        print("  [2]  📋 View Prediction History")
        print("  [3]  🗑  Clear History")
        print("  [4]  ❌ Exit")
        print(Fore.CYAN + "─"*60)

        choice = input(Fore.WHITE + "  Choose an option (1–4): ").strip()

        if choice == "1":
            run_prediction(model, accuracy, importances)
        elif choice == "2":
            view_history()
        elif choice == "3":
            clear_history()
        elif choice == "4":
            print(Fore.CYAN + "\n  👋 Goodbye! Keep studying smart!\n")
            break
        else:
            print(Fore.RED + "  ⚠  Invalid option. Choose 1–4.\n")


# ─────────────────────────────────────────────────────────────
#  Entry point — Python runs this when you start the program.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
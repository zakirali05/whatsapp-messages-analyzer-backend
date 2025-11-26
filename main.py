# main.py

from io import BytesIO
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 👈 force non-GUI backend for server

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from wordcloud import WordCloud
import emoji

from whatsapp_analyzer import process_whatsapp_text

# =========================
# FastAPI app & CORS
# =========================

app = FastAPI(title="WhatsApp Chat Analyzer (Image Charts)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global DataFrame for current uploaded chat
GLOBAL_DF: Optional[pd.DataFrame] = None

# Patterns / constants
MEDIA_PATTERNS = ["<Media omitted>", "Media omitted"]
URL_RE = re.compile(r"https?://\S+")
STOPWORDS = set([
    "the", "and", "are", "you", "for", "with", "this", "that", "have", "not",
    "but", "was", "then", "kya", "hai", "haa", "han", "yes", "your", "our",
    "mein", "hum", "bro", "bhai", "haii"
])

# =========================
# Helper functions
# =========================

def make_dark_theme():
    """Set global dark look for matplotlib / seaborn plots."""
    plt.style.use("dark_background")
    sns.set_style("darkgrid", {
        "axes.facecolor": "#020617",
        "figure.facecolor": "#020617",
        "grid.color": "#1f2933",
    })
    plt.rcParams.update({
        "axes.labelcolor": "#cbd5f5",
        "xtick.color": "#cbd5f5",
        "ytick.color": "#cbd5f5",
        "text.color": "#e5e7eb",
        "axes.edgecolor": "#1f2933",
    })


def fig_to_response(fig) -> StreamingResponse:
    """Convert a matplotlib figure to PNG StreamingResponse."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")


def get_df_or_400() -> pd.DataFrame:
    global GLOBAL_DF
    if GLOBAL_DF is None:
        raise HTTPException(status_code=400, detail="Upload chat first")
    return GLOBAL_DF


def extract_emojis(s: str) -> list[str]:
    return [ch for ch in s if ch in emoji.EMOJI_DATA]


# =========================
# Basic endpoints
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-chat")
async def upload_chat(file: UploadFile = File(...)):
    """
    Upload exported WhatsApp .txt file and parse it into a DataFrame.
    Stores the result in GLOBAL_DF for later chart endpoints.
    """
    global GLOBAL_DF
    try:
        text = (await file.read()).decode("utf-8", errors="ignore")
        df = process_whatsapp_text(text)
        GLOBAL_DF = df
        return {"status": "ok", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Chart endpoints
# =========================

@app.get("/plot/messages-per-user")
def plot_messages_per_user():
    df = get_df_or_400()
    make_dark_theme()

    # Build clean DataFrame with known column names
    user_counts = df["user"].value_counts().reset_index()
    user_counts.columns = ["user", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=user_counts,
        y="user",
        x="count",
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Most Active Users")
    ax.set_xlabel("Messages")
    ax.set_ylabel("User")

    return fig_to_response(fig)



@app.get("/plot/messages-per-hour")
def plot_messages_per_hour():
    df = get_df_or_400()
    make_dark_theme()

    hour_counts = df["hour"].value_counts().sort_index().reset_index()
    hour_counts.columns = ["hour", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=hour_counts,
        x="hour",
        y="count",
        ax=ax,
        color="#38bdf8",
    )
    ax.set_title("Most Active Hours")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Messages")

    return fig_to_response(fig)



@app.get("/plot/messages-per-weekday")
def plot_messages_per_weekday():
    df = get_df_or_400()
    make_dark_theme()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    day_counts = (
        df["weekday"]
        .value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
    )
    day_counts.columns = ["weekday", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=day_counts,
        x="weekday",
        y="count",
        ax=ax,
        color="#a855f7",
    )
    ax.set_title("Activity by Day of Week")
    ax.set_xlabel("")
    ax.set_ylabel("Messages")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    return fig_to_response(fig)



@app.get("/plot/messages-per-month")
def plot_messages_per_month():
    df = get_df_or_400()
    make_dark_theme()

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    month_counts = (
        df["month"]
        .value_counts()
        .reindex(month_order, fill_value=0)
        .reset_index()
    )
    month_counts.columns = ["month", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=month_counts,
        x="month",
        y="count",
        ax=ax,
        marker="o",
        color="#22c55e",
    )
    ax.fill_between(range(len(month_counts)), month_counts["count"], color="#22c55e", alpha=0.2)
    ax.set_title("Most Active Months")
    ax.set_xlabel("")
    ax.set_ylabel("Messages")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    return fig_to_response(fig)



@app.get("/plot/message-timeline")
def plot_message_timeline():
    df = get_df_or_400()
    make_dark_theme()

    daily = (
        df.groupby(df["dates"].dt.date)
          .size()
          .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=daily, x="dates", y="count", ax=ax, color="#e879f9")
    ax.set_title("Message Activity Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Messages")

    return fig_to_response(fig)


@app.get("/plot/heatmap-day-hour")
def plot_heatmap_day_hour():
    df = get_df_or_400()
    make_dark_theme()

    pivot = (
        df.groupby(["weekday", "hour"])
          .size()
          .reset_index(name="count")
          .pivot(index="weekday", columns="hour", values="count")
          .fillna(0)
    )

    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, ax=ax, cmap="viridis")
    ax.set_title("Activity Heatmap (Day vs Hour)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Weekday")

    return fig_to_response(fig)


@app.get("/plot/message-lengths")
def plot_message_lengths():
    df = get_df_or_400()
    make_dark_theme()

    lengths = df["message"].fillna("").str.len()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(lengths, bins=30, ax=ax, color="#38bdf8")
    ax.set_title("Message Length Distribution")
    ax.set_xlabel("Characters per message")
    ax.set_ylabel("Count")

    return fig_to_response(fig)


@app.get("/plot/media-vs-text")
def plot_media_vs_text():
    df = get_df_or_400()
    make_dark_theme()

    is_media = df["message"].fillna("").str.contains(
        "|".join(MEDIA_PATTERNS), case=False
    )
    media_count = int(is_media.sum())
    text_count = int((~is_media).sum())

    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Text messages", "Media messages"]
    sizes = [text_count, media_count]
    colors = ["#38bdf8", "#22c55e"]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"color": "#e5e7eb"},
    )
    ax.set_title("Message Type Distribution")

    return fig_to_response(fig)


@app.get("/plot/links-overall")
def plot_links_overall():
    df = get_df_or_400()
    make_dark_theme()

    has_link = df["message"].fillna("").str.contains(URL_RE)
    links = int(has_link.sum())
    no_links = int((~has_link).sum())

    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Messages without links", "Messages with links"]
    sizes = [no_links, links]
    colors = ["#64748b", "#f97316"]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"color": "#e5e7eb"},
    )
    ax.set_title("Messages Containing Links")

    return fig_to_response(fig)


@app.get("/plot/links-per-user")
def plot_links_per_user():
    df = get_df_or_400()
    make_dark_theme()

    has_link = df["message"].fillna("").str.contains(URL_RE)
    link_df = df[has_link]

    if link_df.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No links found", ha="center", va="center")
        ax.axis("off")
        return fig_to_response(fig)

    user_counts = link_df["user"].value_counts().reset_index()
    user_counts.columns = ["user", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=user_counts,
        y="user",
        x="count",
        ax=ax,
        color="#f97316",
    )
    ax.set_title("Links Shared by User")
    ax.set_xlabel("Links")
    ax.set_ylabel("User")

    return fig_to_response(fig)



@app.get("/plot/top-words")
def plot_top_words():
    df = get_df_or_400()
    make_dark_theme()

    text = " ".join(df["message"].fillna("")).lower()
    text = re.sub(URL_RE, " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    words = [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]
    if not words:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Not enough text for word stats", ha="center", va="center")
        ax.axis("off")
        return fig_to_response(fig)

    counts = pd.Series(words).value_counts().head(20).reset_index()
    counts.columns = ["word", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=counts, x="count", y="word", ax=ax, color="#22c55e")
    ax.set_title("Top 20 Words")
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")

    return fig_to_response(fig)


@app.get("/plot/wordcloud")
def plot_wordcloud():
    df = get_df_or_400()
    make_dark_theme()

    text = " ".join(df["message"].fillna("")).lower()
    text = re.sub(URL_RE, " ", text)

    if not text.strip():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No text for word cloud", ha="center", va="center")
        ax.axis("off")
        return fig_to_response(fig)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="#020617",
        colormap="viridis",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")

    return fig_to_response(fig)


@app.get("/plot/top-emojis")
def plot_top_emojis():
    df = get_df_or_400()
    make_dark_theme()

    all_emojis: list[str] = []
    for msg in df["message"].fillna(""):
        all_emojis.extend(extract_emojis(msg))

    if not all_emojis:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No emojis found", ha="center", va="center")
        ax.axis("off")
        return fig_to_response(fig)

    counts = pd.Series(all_emojis).value_counts().head(20).reset_index()
    counts.columns = ["emoji", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=counts, x="emoji", y="count", ax=ax, color="#facc15")
    ax.set_title("Top Emojis")
    ax.set_xlabel("Emoji")
    ax.set_ylabel("Count")

    return fig_to_response(fig)

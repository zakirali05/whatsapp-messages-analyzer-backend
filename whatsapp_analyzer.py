# whatsapp_analyzer.py
import re
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

pattern = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}:\d{2})\s*([APap][Mm])\]\s([^:]+):\s(.*)$'
)

def process_whatsapp_text(raw_text: str) -> pd.DataFrame:
    lines = raw_text.splitlines()

    datetimes_24 = []
    user_messages = []
    users = []
    messages = []

    for line in lines:
        m = pattern.match(line)
        if not m:
            continue

        date_str, time_str, ampm, user, message = m.groups()
        dt_input = f"{date_str} {time_str} {ampm}"
        dt_obj = datetime.strptime(dt_input, "%d/%m/%y %I:%M:%S %p")
        dt_24 = dt_obj.strftime("%d/%m/%y %H:%M:%S")

        datetimes_24.append(dt_24)
        user_messages.append(f"{user}: {message}")
        users.append(user)
        messages.append(message)

    df = pd.DataFrame({"user_message": user_messages, "message_date": datetimes_24})
    df["message_date"] = pd.to_datetime(df["message_date"], format="%d/%m/%y %H:%M:%S")
    df.rename(columns={"message_date": "dates"}, inplace=True)

    df[["user", "message"]] = df["user_message"].str.split(pat=":", n=1, expand=True)
    df["user"] = df["user"].str.strip()
    df["message"] = df["message"].str.strip()
    df.drop(columns=["user_message"], inplace=True)

    df["year"] = df["dates"].dt.year
    df["month"] = df["dates"].dt.month_name()
    df["day"] = df["dates"].dt.day
    df["hour"] = df["dates"].dt.hour
    df["minute"] = df["dates"].dt.minute
    df["weekday"] = df["dates"].dt.day_name()

    return df

import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET


FPL_BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "FPL Scout/1.0 (+https://streamlit.io)"}


@st.cache_data(ttl=900, show_spinner=False)
def get_fpl_data():
    bootstrap = requests.get(f"{FPL_BASE}/bootstrap-static/", headers=HEADERS, timeout=25)
    bootstrap.raise_for_status()
    fixtures = requests.get(f"{FPL_BASE}/fixtures/", headers=HEADERS, timeout=25)
    fixtures.raise_for_status()
    return bootstrap.json(), fixtures.json()


def team_map(teams: pd.DataFrame) -> Dict[int, str]:
    return dict(zip(teams["id"], teams["name"]))


def normalize_news_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("&#39;", "'")).strip()


@st.cache_data(ttl=900, show_spinner=False)
def web_search(query: str, max_items: int = 8) -> List[Dict[str, str]]:
    """Internet search using Google News RSS, no API key required."""
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-GB&gl=GB&ceid=GB:en"
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    results = []
    for item in root.findall("./channel/item")[:max_items]:
        results.append(
            {
                "title": normalize_news_title(item.findtext("title", "")),
                "url": item.findtext("link", ""),
                "published": item.findtext("pubDate", ""),
                "source": item.findtext("source", ""),
            }
        )
    return results


def add_news_signal(players: pd.DataFrame) -> pd.DataFrame:
    signals = []
    headlines = []
    for _, row in players.iterrows():
        query = f"{row['web_name']} {row['team_name']} injury lineup news Premier League"
        try:
            items = web_search(query, max_items=5)
        except Exception:
            items = []
        text = " ".join(x["title"].lower() for x in items)
        negative = sum(
            text.count(word)
            for word in ["injury", "injured", "doubt", "suspended", "benched", "rested", "out"]
        )
        positive = sum(
            text.count(word)
            for word in ["fit", "returns", "start", "starting", "boost", "available", "likely"]
        )
        signal = max(-2.0, min(2.0, 0.45 * (positive - negative)))
        signals.append(signal)
        headlines.append(" | ".join(x["title"] for x in items[:3]))
    players = players.copy()
    players["news_signal"] = signals
    players["news_headlines"] = headlines
    return players


def prepare_players(bootstrap: dict, fixtures: list, with_news: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name"]]
    players = players.merge(teams[["id", "name", "short_name"]], left_on="team", right_on="id", suffixes=("", "_team"))
    players = players.merge(positions, left_on="element_type", right_on="id", suffixes=("", "_pos"))
    players["team_name"] = players["name"]
    players["position"] = players["singular_name"]
    players["price"] = players["now_cost"] / 10
    players["availability"] = players["chance_of_playing_next_round"].fillna(100)
    players["form_num"] = pd.to_numeric(players["form"], errors="coerce").fillna(0)
    players["minutes_rate"] = (players["minutes"] / (players["starts"].replace(0, 1))).clip(0, 90)
    players["base_score"] = (
        players["ep_next"].fillna(0)
        + 0.20 * players["form_num"]
        + 0.015 * players["selected_by_percent"].fillna(0)
        + 0.008 * players["minutes_rate"].fillna(0)
    )
    if with_news:
        players = add_news_signal(players)
    else:
        players["news_signal"] = 0.0
        players["news_headlines"] = ""
    players["predicted_points"] = (
        players["base_score"]
        * (players["availability"] / 100)
        + players["news_signal"]
    ).clip(lower=0)
    fixtures_df = pd.DataFrame(fixtures)
    return players, fixtures_df


def fixture_adjustments(players: pd.DataFrame, fixtures: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if fixtures.empty:
        players["fixture_score"] = 0.0
        players["fixture_summary"] = "Unavailable"
        return players
    teams = pd.DataFrame({"team": players["team"].unique()})
    rows = []
    for team_id in teams["team"]:
        upcoming = fixtures[
            (fixtures["event"].notna())
            & (fixtures["event"] >= 1)
            & (fixtures["event"] <= horizon)
            & ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id))
        ].sort_values("event")
        scores = []
        labels = []
        for _, fixture in upcoming.iterrows():
            is_home = fixture["team_h"] == team_id
            difficulty = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
            scores.append(6 - float(difficulty))
            opp_id = fixture["team_a"] if is_home else fixture["team_h"]
            labels.append(f"{'H' if is_home else 'A'} vs {opp_id} ({int(difficulty)})")
        rows.append(
            {
                "team": team_id,
                "fixture_score": sum(scores) / max(1, len(scores)),
                "fixture_summary": ", ".join(labels[:horizon]),
            }
        )
    fixture_df = pd.DataFrame(rows)
    players = players.merge(fixture_df, on="team", how="left")
    players["fixture_score"] = players["fixture_score"].fillna(0)
    players["predicted_points"] = players["predicted_points"] + 0.45 * players["fixture_score"]
    return players


def greedy_squad(players: pd.DataFrame, budget: float, horizon: int) -> pd.DataFrame:
    # A transparent optimizer: enforce FPL squad composition, budget and max three per club.
    limits = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
    selected = []
    spent = 0.0
    club_counts: Dict[int, int] = {}
    for position, count in limits.items():
        pool = players[players["position"] == position].sort_values(
            ["predicted_points", "value_score"], ascending=False
        )
        for _, player in pool.iterrows():
            if len([x for x in selected if x["position"] == position]) >= count:
                break
            if spent + player["price"] > budget:
                continue
            if club_counts.get(int(player["team"]), 0) >= 3:
                continue
            selected.append(player)
            spent += float(player["price"])
            club_counts[int(player["team"])] = club_counts.get(int(player["team"]), 0) + 1
    squad = pd.DataFrame(selected)
    if len(squad) < 15:
        # Fill any gaps with cheapest available players while preserving constraints.
        remaining = players[~players["id"].isin(squad["id"] if not squad.empty else [])].sort_values("price")
        for _, player in remaining.iterrows():
            if len(squad) >= 15 or spent + player["price"] > budget:
                continue
            pos_count = (squad["position"] == player["position"]).sum()
            if pos_count >= limits.get(player["position"], 0) or club_counts.get(int(player["team"]), 0) >= 3:
                continue
            squad = pd.concat([squad, pd.DataFrame([player])], ignore_index=True)
            spent += float(player["price"])
            club_counts[int(player["team"])] = club_counts.get(int(player["team"]), 0) + 1
    return squad.sort_values(["position", "predicted_points"], ascending=[True, False])


def pick_xi(squad: pd.DataFrame) -> pd.DataFrame:
    starters = []
    goalkeepers = squad[squad["position"] == "Goalkeeper"].sort_values("predicted_points", ascending=False)
    starters.append(goalkeepers.head(1))
    defenders = squad[squad["position"] == "Defender"].sort_values("predicted_points", ascending=False).head(3)
    starters.append(defenders)
    mids = squad[squad["position"] == "Midfielder"].sort_values("predicted_points", ascending=False).head(4)
    starters.append(mids)
    forwards = squad[squad["position"] == "Forward"].sort_values("predicted_points", ascending=False).head(3)
    starters.append(forwards)
    return pd.concat(starters, ignore_index=True)


def main():
    st.set_page_config(page_title="FPL Scout AI", page_icon="⚽", layout="wide")
    st.title("⚽ FPL Scout AI")
    st.caption("A live Fantasy Premier League research assistant using official FPL data plus web headlines and expert-analysis searches.")

    with st.sidebar:
        st.header("Settings")
        budget = st.number_input("Squad budget (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)
        horizon = st.slider("Fixture horizon", 1, 8, 5)
        use_news = st.checkbox("Search live web/news signals", value=True)
        st.info("Web searches can take a little longer because each player is checked for recent availability and lineup news.")

    try:
        bootstrap, fixtures = get_fpl_data()
    except Exception as exc:
        st.error(f"Could not reach the official FPL API - {exc}")
        st.stop()

    with st.spinner("Loading current FPL data and researching the web..."):
        players, fixtures_df = prepare_players(bootstrap, fixtures, with_news=use_news)
        players = fixture_adjustments(players, fixtures_df, horizon)
        players["value_score"] = players["predicted_points"] / players["price"].replace(0, 99)
        squad = greedy_squad(players, budget, horizon)
        xi = pick_xi(squad)

    st.success(f"Recommended squad found - projected spend £{squad['price'].sum():.1f}m and {len(squad)} players.")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Recommended starting XI")
        show_xi = xi[["web_name", "position", "team_name", "price", "predicted_points", "fixture_summary"]].copy()
        show_xi.columns = ["Player", "Position", "Club", "Price", "Projection", "Upcoming fixtures"]
        st.dataframe(show_xi, use_container_width=True, hide_index=True)
        st.metric("Starting XI projection", f"{xi['predicted_points'].sum():.1f} points")
    with right:
        st.subheader("Squad budget")
        st.metric("Spend", f"£{squad['price'].sum():.1f}m")
        st.metric("Remaining", f"£{budget - squad['price'].sum():.1f}m")
        st.bar_chart(squad.groupby("position")["predicted_points"].sum())

    st.subheader("Best player targets")
    targets = players.sort_values("predicted_points", ascending=False).head(20)
    target_view = targets[
        ["web_name", "position", "team_name", "price", "predicted_points", "form_num", "availability", "news_signal"]
    ].copy()
    target_view.columns = ["Player", "Position", "Club", "Price", "Projection", "Form", "Availability %", "Web signal"]
    st.dataframe(target_view, use_container_width=True, hide_index=True)

    st.subheader("Research notes")
    selected_names = set(xi["web_name"].tolist())
    for _, row in players[players["web_name"].isin(selected_names)].iterrows():
        with st.expander(f"{row['web_name']} - {row['team_name']}"):
            st.write(f"**Projection** {row['predicted_points']:.1f} | **Availability** {row['availability']:.0f}% | **Fixture outlook** {row['fixture_summary']}")
            if row["news_headlines"]:
                st.write(row["news_headlines"])
            else:
                st.write("No recent headlines returned.")

    st.caption(f"Data refreshed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. This is an analytical aid, not financial or betting advice.")


if __name__ == "__main__":
    main()

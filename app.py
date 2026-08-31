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

    players = players.merge(
        teams[["id", "name", "short_name"]],
        left_on="team",
        right_on="id",
        suffixes=("", "_team"),
    )
    players = players.merge(
        positions,
        left_on="element_type",
        right_on="id",
        suffixes=("", "_pos"),
    )

    players["team_name"] = players["name"]
    players["position"] = players["singular_name"]
    players["price"] = pd.to_numeric(players["now_cost"], errors="coerce").fillna(0) / 10

    players["availability"] = pd.to_numeric(
        players["chance_of_playing_next_round"], errors="coerce"
    ).fillna(100).clip(0, 100)

    # FPL can return some numeric-looking fields as strings.
    players["ep_next_num"] = pd.to_numeric(players["ep_next"], errors="coerce").fillna(0)
    players["form_num"] = pd.to_numeric(players["form"], errors="coerce").fillna(0)
    players["selected_by_percent_num"] = pd.to_numeric(
        players["selected_by_percent"], errors="coerce"
    ).fillna(0)
    players["minutes_num"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(0)
    players["starts_num"] = pd.to_numeric(players["starts"], errors="coerce").fillna(0)

    players["minutes_rate"] = (
        players["minutes_num"] / players["starts_num"].replace(0, 1)
    ).clip(0, 90)

    players["base_score"] = (
        players["ep_next_num"]
        + 0.20 * players["form_num"]
        + 0.015 * players["selected_by_percent_num"]
        + 0.008 * players["minutes_rate"]
    )

    if with_news:
        players = add_news_signal(players)
    else:
        players["news_signal"] = 0.0
        players["news_headlines"] = ""

    players["predicted_points"] = (
        players["base_score"] * (players["availability"] / 100)
        + players["news_signal"]
    ).clip(lower=0)

    fixtures_df = pd.DataFrame(fixtures)
    return players, fixtures_df


def fixture_adjustments(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    start_gw: int,
    end_gw: Optional[int] = None,
) -> pd.DataFrame:
    """Add fixture difficulty for one GW or a GW range.

    A single GW gives a focused projection. A range uses the average
    fixture score across the selected Gameweeks.
    """
    players = players.copy()

    if end_gw is None:
        end_gw = start_gw

    if fixtures.empty:
        players["fixture_score"] = 0.0
        players["fixture_summary"] = "Unavailable"
        return players

    fixtures = fixtures.copy()
    fixtures["event"] = pd.to_numeric(fixtures["event"], errors="coerce")

    rows = []
    for team_id in players["team"].dropna().unique():
        upcoming = fixtures[
            (fixtures["event"].notna())
            & (fixtures["event"] >= start_gw)
            & (fixtures["event"] <= end_gw)
            & ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id))
        ].sort_values("event")

        scores = []
        labels = []

        for _, fixture in upcoming.iterrows():
            is_home = fixture["team_h"] == team_id
            difficulty = (
                fixture["team_h_difficulty"]
                if is_home
                else fixture["team_a_difficulty"]
            )
            difficulty = pd.to_numeric(difficulty, errors="coerce")
            if pd.isna(difficulty):
                continue

            scores.append(6 - float(difficulty))

            opp_id = fixture["team_a"] if is_home else fixture["team_h"]
            labels.append(
                f"GW{int(fixture['event'])} {'H' if is_home else 'A'} vs {int(opp_id)} "
                f"({int(difficulty)})"
            )

        rows.append(
            {
                "team": team_id,
                "fixture_score": sum(scores) / max(1, len(scores)),
                "fixture_summary": ", ".join(labels),
            }
        )

    fixture_df = pd.DataFrame(rows)
    players = players.merge(fixture_df, on="team", how="left")
    players["fixture_score"] = players["fixture_score"].fillna(0)
    players["fixture_summary"] = players["fixture_summary"].fillna("Unavailable")

    # 0.45 is deliberately modest so fixtures influence, rather than dominate,
    # the FPL/API form and expected-points signals.
    players["predicted_points"] = (
        players["predicted_points"] + 0.45 * players["fixture_score"]
    ).clip(lower=0)

    return players


def greedy_squad(players: pd.DataFrame, budget: float, horizon: int = 1) -> pd.DataFrame:
    """Transparent FPL squad builder: 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs,
    max three players per club and within the selected budget.
    """
    limits = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
    selected = []
    spent = 0.0
    club_counts: Dict[int, int] = {}

    # A small value component helps prevent the greedy model from spending
    # the whole budget on expensive players too early.
    pool = players.copy()
    pool["value_score"] = (
        pool["predicted_points"] / pool["price"].replace(0, 99)
    )

    for position, count in limits.items():
        pos_pool = pool[pool["position"] == position].sort_values(
            ["predicted_points", "value_score"], ascending=False
        )

        for _, player in pos_pool.iterrows():
            if sum(x["position"] == position for x in selected) >= count:
                break
            if spent + float(player["price"]) > budget:
                continue
            if club_counts.get(int(player["team"]), 0) >= 3:
                continue

            selected.append(player)
            spent += float(player["price"])
            club_counts[int(player["team"])] = (
                club_counts.get(int(player["team"]), 0) + 1
            )

    squad = pd.DataFrame(selected)

    # Fill any gaps with the best affordable remaining player while preserving
    # positional and club constraints.
    remaining = pool[~pool["id"].isin(squad["id"] if not squad.empty else [])].sort_values(
        ["value_score", "predicted_points"], ascending=False
    )

    for _, player in remaining.iterrows():
        if len(squad) >= 15:
            break
        if spent + float(player["price"]) > budget:
            continue

        pos_count = (
            (squad["position"] == player["position"]).sum()
            if not squad.empty
            else 0
        )
        if pos_count >= limits.get(player["position"], 0):
            continue
        if club_counts.get(int(player["team"]), 0) >= 3:
            continue

        squad = pd.concat([squad, pd.DataFrame([player])], ignore_index=True)
        spent += float(player["price"])
        club_counts[int(player["team"])] = club_counts.get(int(player["team"]), 0) + 1

    return squad.sort_values(
        ["position", "predicted_points"], ascending=[True, False]
    ).reset_index(drop=True)


def pick_xi(squad: pd.DataFrame) -> pd.DataFrame:
    starters = []

    goalkeepers = squad[squad["position"] == "Goalkeeper"].sort_values(
        "predicted_points", ascending=False
    )
    starters.append(goalkeepers.head(1))

    defenders = squad[squad["position"] == "Defender"].sort_values(
        "predicted_points", ascending=False
    ).head(3)
    starters.append(defenders)

    mids = squad[squad["position"] == "Midfielder"].sort_values(
        "predicted_points", ascending=False
    ).head(4)
    starters.append(mids)

    forwards = squad[squad["position"] == "Forward"].sort_values(
        "predicted_points", ascending=False
    ).head(3)
    starters.append(forwards)

    return pd.concat(starters, ignore_index=True)


def captain_choices(xi: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Choose captain and vice captain from the starting XI."""
    ranked = xi.sort_values(
        ["predicted_points", "form_num"], ascending=False
    ).reset_index(drop=True)

    captain = ranked.iloc[0]
    vice = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
    return captain, vice


def add_fpl_labels(players: pd.DataFrame, squad: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """Add useful FPL decision labels for the UI."""
    out = players.copy()

    squad_ids = set(squad["id"].tolist()) if not squad.empty else set()
    xi_ids = set(xi["id"].tolist()) if not xi.empty else set()

    out["in_squad"] = out["id"].isin(squad_ids)
    out["starter"] = out["id"].isin(xi_ids)

    # Differential = low ownership but strong projection/value.
    out["differential_score"] = (
        out["predicted_points"]
        / (1 + out["selected_by_percent_num"])
    )

    return out


def player_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "web_name",
        "position",
        "team_name",
        "price",
        "predicted_points",
        "form_num",
        "availability",
        "selected_by_percent_num",
    ]
    view = df[cols].copy()
    view.columns = [
        "Player",
        "Position",
        "Club",
        "Price",
        "Projection",
        "Form",
        "Availability %",
        "Ownership %",
    ]
    return view


def format_player_name(row: pd.Series) -> str:
    return f"{row['web_name']} ({row['team_name']})"


def main():
    st.set_page_config(page_title="FPL Scout AI", page_icon="⚽", layout="wide")
    st.title("⚽ FPL Scout AI")
    st.caption(
        "A live Fantasy Premier League research assistant using official FPL data "
        "plus Google News headlines."
    )

    with st.sidebar:
        st.header("Settings")

        mode = st.radio(
            "Analysis mode",
            ["Single Gameweek", "Multi-Gameweek"],
            help="Single GW focuses on one Gameweek. Multi-GW averages the fixture outlook across a range.",
        )

        # FPL normally has a current/next event. The selector remains flexible
        # so the same app can be used throughout the season.
        max_gw = 38

        if mode == "Single Gameweek":
            selected_gw = st.number_input(
                "Gameweek",
                min_value=1,
                max_value=max_gw,
                value=1,
                step=1,
            )
            start_gw = int(selected_gw)
            end_gw = int(selected_gw)
            st.caption(f"Optimizing specifically for GW{start_gw}.")
        else:
            start_gw = st.number_input(
                "Start Gameweek",
                min_value=1,
                max_value=max_gw,
                value=1,
                step=1,
            )
            end_gw = st.number_input(
                "End Gameweek",
                min_value=int(start_gw),
                max_value=max_gw,
                value=min(int(start_gw) + 5, max_gw),
                step=1,
            )
            start_gw = int(start_gw)
            end_gw = int(end_gw)
            st.caption(f"Optimizing across GW{start_gw}–GW{end_gw}.")

        budget = st.number_input(
            "Squad budget (£m)",
            min_value=80.0,
            max_value=120.0,
            value=100.0,
            step=0.5,
        )

        use_news = st.checkbox(
            "Search live web/news signals",
            value=True,
        )

        ownership_limit = st.slider(
            "Differential ownership ceiling %",
            min_value=1.0,
            max_value=15.0,
            value=8.0,
            step=1.0,
        )

        st.info(
            "Live news searches can take longer because players are checked for "
            "recent injury, suspension and lineup information."
        )

    try:
        bootstrap, fixtures = get_fpl_data()
    except Exception as exc:
        st.error(f"Could not reach the official FPL API - {exc}")
        st.stop()

    # Use the API's next/current Gameweek when possible to make the default
    # selection more useful during the season.
    events = pd.DataFrame(bootstrap.get("events", []))
    if not events.empty:
        current_candidates = events[
            events["finished"].eq(False)
            & events["is_previous"].eq(False)
        ]
        if not current_candidates.empty:
            next_gw = int(current_candidates.iloc[0]["id"])
            if mode == "Single Gameweek" and start_gw == 1 and next_gw > 1:
                st.info(
                    f"The next unfinished FPL Gameweek appears to be GW{next_gw}. "
                    "You can select it in the sidebar."
                )

    with st.spinner("Loading current FPL data and researching the web..."):
        players, fixtures_df = prepare_players(
            bootstrap,
            fixtures,
            with_news=use_news,
        )
        players = fixture_adjustments(
            players,
            fixtures_df,
            start_gw,
            end_gw,
        )
        players["value_score"] = (
            players["predicted_points"]
            / players["price"].replace(0, 99)
        )

        squad = greedy_squad(players, budget, end_gw - start_gw + 1)

        if squad.empty or len(squad) < 15:
            st.error(
                "The optimizer could not build a complete 15-player squad within "
                "the selected budget. Try increasing the budget."
            )
            st.stop()

        xi = pick_xi(squad)
        captain, vice = captain_choices(xi)
        players = add_fpl_labels(players, squad, xi)

    # Top-level summary.
    st.success(
        f"Recommended squad found — projected spend £{squad['price'].sum():.1f}m "
        f"and {len(squad)} players."
    )

    captain_col, vice_col, diff_col, avoid_col = st.columns(4)

    with captain_col:
        st.metric(
            "🎯 Captain",
            captain["web_name"],
            f"{captain['predicted_points']:.1f} projected",
        )

    with vice_col:
        st.metric(
            "🥈 Vice Captain",
            vice["web_name"],
            f"{vice['predicted_points']:.1f} projected",
        )

    differentials = players[
        (players["selected_by_percent_num"] <= ownership_limit)
        & (~players["in_squad"])
    ].sort_values(
        ["differential_score", "predicted_points"],
        ascending=False,
    )

    with diff_col:
        if not differentials.empty:
            diff = differentials.iloc[0]
            st.metric(
                "🔥 Best Differential",
                diff["web_name"],
                f"{diff['selected_by_percent_num']:.1f}% owned",
            )
        else:
            st.metric("🔥 Best Differential", "None found")

    avoid = players.sort_values(
        ["predicted_points", "availability"],
        ascending=[True, True],
    )
    avoid = avoid[avoid["price"] >= 4.5]

    with avoid_col:
        if not avoid.empty:
            avoid_player = avoid.iloc[0]
            st.metric(
                "⚠️ Avoid",
                avoid_player["web_name"],
                f"{avoid_player['predicted_points']:.1f} projected",
            )
        else:
            st.metric("⚠️ Avoid", "None found")

    st.divider()

    left, right = st.columns([1.5, 1])

    with left:
        st.subheader(
            "Recommended starting XI"
            if mode == "Single Gameweek"
            else f"Recommended XI for GW{start_gw}–GW{end_gw}"
        )

        show_xi = xi[
            [
                "web_name",
                "position",
                "team_name",
                "price",
                "predicted_points",
                "fixture_summary",
            ]
        ].copy()

        show_xi.columns = [
            "Player",
            "Position",
            "Club",
            "Price",
            "Projection",
            "Fixture outlook",
        ]

        st.dataframe(
            show_xi,
            use_container_width=True,
            hide_index=True,
        )
        st.metric(
            "Starting XI projection",
            f"{xi['predicted_points'].sum():.1f} points",
        )

    with right:
        st.subheader("Squad budget")
        st.metric("Spend", f"£{squad['price'].sum():.1f}m")
        st.metric(
            "Remaining",
            f"£{budget - squad['price'].sum():.1f}m",
        )

        st.subheader("Captain plan")
        st.write(
            f"**Captain:** {format_player_name(captain)} — "
            f"{captain['predicted_points']:.1f} projection"
        )
        st.write(
            f"**Vice:** {format_player_name(vice)} — "
            f"{vice['predicted_points']:.1f} projection"
        )

        st.bar_chart(
            squad.groupby("position")["predicted_points"].sum()
        )

    st.divider()

    st.subheader("🔥 Best differentials")
    if differentials.empty:
        st.info("No strong low-ownership differential was found.")
    else:
        diff_view = player_table(differentials.head(10))
        st.dataframe(diff_view, use_container_width=True, hide_index=True)

    st.subheader("💰 Best budget options")
    budget_options = players[
        (players["price"] <= 6.0)
        & (~players["in_squad"])
    ].sort_values(
        ["value_score", "predicted_points"],
        ascending=False,
    )
    if budget_options.empty:
        st.info("No additional budget options found.")
    else:
        st.dataframe(
            player_table(budget_options.head(10)),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("⚠️ Players to avoid")
    avoid_view = players.sort_values(
        ["predicted_points", "availability"],
        ascending=[True, True],
    )
    avoid_view = avoid_view[
        (avoid_view["price"] >= 4.5)
        & (~avoid_view["in_squad"])
    ].head(10)

    if avoid_view.empty:
        st.info("No obvious avoid candidates found.")
    else:
        st.dataframe(
            player_table(avoid_view),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📋 Full recommended squad")
    squad_view = player_table(squad)
    st.dataframe(
        squad_view,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("📰 Research notes")
    selected_names = set(xi["web_name"].tolist())

    for _, row in players[
        players["web_name"].isin(selected_names)
    ].iterrows():
        with st.expander(
            f"{row['web_name']} - {row['team_name']}"
        ):
            st.write(
                f"**Projection:** {row['predicted_points']:.1f} | "
                f"**Availability:** {row['availability']:.0f}% | "
                f"**Fixture outlook:** {row['fixture_summary']}"
            )

            if row["news_headlines"]:
                st.write(row["news_headlines"])
            else:
                st.write("No recent headlines returned.")

    st.caption(
        f"Analysis: {'GW' + str(start_gw) if start_gw == end_gw else f'GW{start_gw}–GW{end_gw}'}. "
        f"Data refreshed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        "This is an analytical aid, not financial or betting advice."
    )


if __name__ == "__main__":
    main()

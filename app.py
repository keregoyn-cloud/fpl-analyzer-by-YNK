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


@st.cache_data(ttl=1800, show_spinner=False)
def get_player_history(player_id: int) -> dict:
    """Fetch one player's season history for home/away and per-90 diagnostics."""
    response = requests.get(
        f"{FPL_BASE}/element-summary/{int(player_id)}/",
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=1800, show_spinner=False)
def get_advanced_player_data(player_ids: tuple) -> pd.DataFrame:
    """Build historical home/away and attacking/defensive signals."""
    rows = []
    for player_id in player_ids:
        try:
            payload = get_player_history(int(player_id))
            history = pd.DataFrame(payload.get("history", []))
            if history.empty:
                continue

            for col in [
                "minutes", "goals_scored", "assists", "clean_sheets",
                "goals_conceded", "saves", "bonus", "bps", "influence",
                "creativity", "threat", "ict_index", "expected_goals",
                "expected_assists", "total_points", "starts",
            ]:
                if col not in history.columns:
                    history[col] = 0
                history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)

            history["was_home"] = history["was_home"].astype(bool)
            home = history[history["was_home"]]
            away = history[~history["was_home"]]

            def p90(frame, col):
                mins = max(float(frame["minutes"].sum()), 1.0)
                return float(frame[col].sum()) / mins * 90.0

            rows.append({
                "id": int(player_id),
                "hist_games": int(len(history)),
                "home_ppg": float(home["total_points"].mean()) if not home.empty else 0.0,
                "away_ppg": float(away["total_points"].mean()) if not away.empty else 0.0,
                "home_xg90": p90(home, "expected_goals") if not home.empty else 0.0,
                "away_xg90": p90(away, "expected_goals") if not away.empty else 0.0,
                "home_xa90": p90(home, "expected_assists") if not home.empty else 0.0,
                "away_xa90": p90(away, "expected_assists") if not away.empty else 0.0,
                "xg90_hist": p90(history, "expected_goals"),
                "xa90_hist": p90(history, "expected_assists"),
                "goals90_hist": p90(history, "goals_scored"),
                "assists90_hist": p90(history, "assists"),
                "saves90_hist": p90(history, "saves"),
                "bps90_hist": p90(history, "bps"),
                "clean_sheet_rate_hist": float(history["clean_sheets"].sum() / max(len(history), 1)),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


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


def prepare_players(bootstrap: dict, fixtures: list, with_news: bool = True, advanced_history: bool = False, history_limit: int = 250) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Set-piece roles available from the official FPL player feed.
    for col in [
        "penalties_order",
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
    ]:
        if col not in players.columns:
            players[col] = pd.NA
        players[col] = pd.to_numeric(players[col], errors="coerce")

    players["set_piece_score"] = (
        players["penalties_order"].notna().astype(float)
        + 0.55 * players["corners_and_indirect_freekicks_order"].notna().astype(float)
        + 0.35 * players["direct_freekicks_order"].notna().astype(float)
    ).clip(0, 1.5)

    # Conservative expected-minutes estimate.
    players["expected_minutes"] = (
        90.0 * (players["availability"] / 100.0)
        * (0.45 + 0.55 * (players["minutes_rate"] / 90.0))
    ).clip(0, 90)

    # Defaults when deep history is disabled.
    players["advanced_history_loaded"] = False
    players["home_ppg"] = players["form_num"]
    players["away_ppg"] = players["form_num"]
    players["home_xg90"] = 0.0
    players["away_xg90"] = 0.0
    players["home_xa90"] = 0.0
    players["away_xa90"] = 0.0
    players["xg90_hist"] = 0.0
    players["xa90_hist"] = 0.0
    players["goals90_hist"] = 0.0
    players["assists90_hist"] = 0.0
    players["saves90_hist"] = 0.0
    players["bps90_hist"] = 0.0
    players["clean_sheet_rate_hist"] = 0.0
    players["hist_games"] = 0

    if advanced_history:
        ids = tuple(
            players.sort_values(
                ["ep_next_num", "form_num", "minutes_num"],
                ascending=False,
            )["id"].head(int(history_limit)).astype(int).tolist()
        )
        hist = get_advanced_player_data(ids)
        if not hist.empty:
            players = players.merge(hist, on="id", how="left", suffixes=("", "_advanced"))
            for col in [
                "hist_games", "home_ppg", "away_ppg", "home_xg90", "away_xg90",
                "home_xa90", "away_xa90", "xg90_hist", "xa90_hist",
                "goals90_hist", "assists90_hist", "saves90_hist",
                "bps90_hist", "clean_sheet_rate_hist",
            ]:
                alt = f"{col}_advanced"
                if alt in players.columns:
                    players[col] = players[alt].fillna(players[col])
                    players.drop(columns=[alt], inplace=True)
            players["advanced_history_loaded"] = players["hist_games"].fillna(0).gt(0)

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



def build_fixture_projections(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    start_gw: int,
    end_gw: int,
) -> pd.DataFrame:
    """Create player-level expected points for every Gameweek in the selected range.

    This is a transparent model rather than a claim to reproduce FPL's proprietary
    expected-points model. It uses the current FPL signals in the API, fixture FDR,
    home/away, availability, form, minutes reliability and live-news signal.
    """
    players = players.copy()
    fixtures = fixtures.copy()

    if fixtures.empty:
        rows = []
        for _, p in players.iterrows():
            for gw in range(start_gw, end_gw + 1):
                rows.append(
                    {
                        "id": int(p["id"]),
                        "gw": gw,
                        "expected_points": 0.0,
                        "fixture_count": 0,
                        "fixture_summary": "No fixture data",
                    }
                )
        return pd.DataFrame(rows)

    fixtures["event"] = pd.to_numeric(fixtures["event"], errors="coerce")
    fixtures["team_h"] = pd.to_numeric(fixtures["team_h"], errors="coerce")
    fixtures["team_a"] = pd.to_numeric(fixtures["team_a"], errors="coerce")
    fixtures["team_h_difficulty"] = pd.to_numeric(
        fixtures["team_h_difficulty"], errors="coerce"
    )
    fixtures["team_a_difficulty"] = pd.to_numeric(
        fixtures["team_a_difficulty"], errors="coerce"
    )

    team_names = dict(zip(players["team"], players["team_name"]))
    rows = []

    # Difficulty multipliers are deliberately modest: fixtures should matter,
    # but should not overwhelm player quality.
    fdr_multiplier = {
        1: 1.16,
        2: 1.08,
        3: 1.00,
        4: 0.91,
        5: 0.82,
    }

    for _, p in players.iterrows():
        team_id = p["team"]
        player_base = max(
            1.5,
            float(p["base_score"]) + 0.20 * float(p.get("news_signal", 0.0)),
        )
        availability_factor = max(0.0, min(1.0, float(p["availability"]) / 100.0))

        for gw in range(start_gw, end_gw + 1):
            gw_fixtures = fixtures[
                (fixtures["event"] == gw)
                & (
                    (fixtures["team_h"] == team_id)
                    | (fixtures["team_a"] == team_id)
                )
            ]

            total_points = 0.0
            labels = []

            for _, fx in gw_fixtures.iterrows():
                is_home = fx["team_h"] == team_id
                difficulty = (
                    fx["team_h_difficulty"]
                    if is_home
                    else fx["team_a_difficulty"]
                )
                if pd.isna(difficulty):
                    difficulty = 3

                difficulty = int(max(1, min(5, difficulty)))
                opponent = fx["team_a"] if is_home else fx["team_h"]
                opponent_name = team_names.get(opponent, f"Team {int(opponent)}")

                home_factor = 1.04 if is_home else 0.97
                fdr_factor = fdr_multiplier.get(difficulty, 1.0)

                # The base model is adjusted per fixture, so DGWs naturally
                # receive two opportunities and BGWs receive zero.
                fixture_points = (
                    player_base
                    * availability_factor
                    * home_factor
                    * fdr_factor
                )

                # Appearance reliability matters more for future planning.
                minutes_factor = min(
                    1.0,
                    max(0.55, float(p["minutes_rate"]) / 90.0),
                )
                fixture_points *= 0.75 + 0.25 * minutes_factor

                # Historical home/away performance.
                venue_ppg = float(p["home_ppg"] if is_home else p["away_ppg"])
                current_form = max(float(p["form_num"]), 1.0)
                if venue_ppg > 0:
                    venue_ratio = max(0.90, min(1.10, 0.94 + 0.06 * venue_ppg / current_form))
                    fixture_points *= venue_ratio

                # Player-specific attacking involvement and set pieces.
                xg90 = float(p["home_xg90"] if is_home else p["away_xg90"])
                xa90 = float(p["home_xa90"] if is_home else p["away_xa90"])
                attack_boost = min(
                    0.16,
                    0.07 * xg90 + 0.04 * xa90 + 0.03 * float(p["set_piece_score"]),
                )
                fixture_points *= 1.0 + max(0.0, attack_boost)

                # Defensive clean-sheet probability proxy.
                if p["position"] in {"Goalkeeper", "Defender"}:
                    fdr_cs = {1: 1.10, 2: 1.06, 3: 1.00, 4: 0.93, 5: 0.86}.get(difficulty, 1.0)
                    hist_cs = float(p["clean_sheet_rate_hist"])
                    cs_factor = 0.94 + 0.06 * fdr_cs + min(0.04, hist_cs * 0.04)
                    fixture_points *= cs_factor

                # Expected minutes have a capped effect on every fixture.
                fixture_points *= 0.65 + 0.35 * (float(p["expected_minutes"]) / 90.0)

                total_points += fixture_points

                labels.append(
                    f"{'H' if is_home else 'A'} vs {opponent_name} "
                    f"(FDR {difficulty})"
                )

            rows.append(
                {
                    "id": int(p["id"]),
                    "gw": gw,
                    "expected_points": round(total_points, 2),
                    "fixture_count": len(gw_fixtures),
                    "fixture_summary": (
                        ", ".join(labels) if labels else "BLANK"
                    ),
                }
            )

    return pd.DataFrame(rows)


def apply_fixture_projections(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    start_gw: int,
    end_gw: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach per-GW projections and aggregate projections to players."""
    projections = build_fixture_projections(
        players,
        fixtures,
        start_gw,
        end_gw,
    )

    agg = (
        projections.groupby("id", as_index=False)
        .agg(
            projection_total=("expected_points", "sum"),
            projection_avg=("expected_points", "mean"),
            fixture_count=("fixture_count", "sum"),
        )
    )

    players = players.drop(
        columns=[
            c
            for c in [
                "predicted_points",
                "fixture_score",
                "fixture_summary",
            ]
            if c in players.columns
        ],
        errors="ignore",
    )

    players = players.merge(agg, on="id", how="left")
    players["projection_total"] = players["projection_total"].fillna(0.0)
    players["projection_avg"] = players["projection_avg"].fillna(0.0)
    players["fixture_count"] = players["fixture_count"].fillna(0).astype(int)

    # For a single GW this is that GW's projection. For a range it is the
    # total projected return across the range, which is useful for squad
    # planning.
    players["predicted_points"] = players["projection_total"]

    # Useful for ranking captaincy and transfer targets.
    players["captain_score"] = (
        players["projection_avg"] * 0.78
        + players["form_num"] * 0.12
        + players["availability"] * 0.10
    )

    players["value_score"] = (
        players["projection_total"]
        / players["price"].replace(0, 99)
    )

    return players, projections


def price_change_signal(players: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent market/price-change heuristic.

    FPL's exact price-change formula is not public. This therefore reports
    observed transfer momentum and recent price movement, not an official
    prediction.
    """
    players = players.copy()

    for col in [
        "transfers_in_event",
        "transfers_out_event",
        "cost_change_event",
        "cost_change_start",
        "cost_change_start_fall",
    ]:
        if col not in players.columns:
            players[col] = 0
        players[col] = pd.to_numeric(
            players[col], errors="coerce"
        ).fillna(0)

    players["net_transfer_event"] = (
        players["transfers_in_event"]
        - players["transfers_out_event"]
    )

    # Normalize by ownership to avoid making raw transfer counts alone
    # dominate the signal.
    players["transfer_momentum"] = (
        players["net_transfer_event"]
        / (players["selected_by_percent_num"] + 1.0)
    )

    def label(row):
        event_change = row["cost_change_event"]
        momentum = row["transfer_momentum"]

        if event_change > 0:
            return "Rising"
        if event_change < 0:
            return "Falling"
        if momentum > 5000:
            return "Likely rise pressure"
        if momentum < -5000:
            return "Likely fall pressure"
        return "Stable"

    players["price_trend"] = players.apply(label, axis=1)

    # Conservative projected market value for strategy display only.
    players["price_trend_score"] = players["transfer_momentum"].clip(-20000, 20000) / 20000
    players["strategy_price"] = (
        players["price"] + players["cost_change_event"] * 0.1
    ).clip(lower=3.5)

    return players


def greedy_squad(
    players: pd.DataFrame,
    budget: float,
    horizon: int = 1,
) -> pd.DataFrame:
    """Build a legal FPL 15-man squad.

    Rules enforced:
    - 2 goalkeepers
    - 5 defenders
    - 5 midfielders
    - 3 forwards
    - maximum 3 players from one club
    - budget limit
    """
    limits = {
        "Goalkeeper": 2,
        "Defender": 5,
        "Midfielder": 5,
        "Forward": 3,
    }

    pool = players.copy()
    pool["selection_score"] = (
        pool["projection_total"]
        + 0.18 * pool["value_score"]
        + 0.15 * pool["captain_score"]
        + 0.05 * pool["price_trend_score"]
    )

    selected = []
    spent = 0.0
    club_counts: Dict[int, int] = {}

    # First create a valid positional skeleton, prioritizing projection but
    # retaining enough budget to complete the squad.
    for position, count in limits.items():
        pos_pool = pool[
            pool["position"] == position
        ].sort_values(
            ["selection_score", "value_score"],
            ascending=False,
        )

        for _, player in pos_pool.iterrows():
            if sum(x["position"] == position for x in selected) >= count:
                break

            price = float(player["price"])
            club = int(player["team"])

            if spent + price > budget:
                continue
            if club_counts.get(club, 0) >= 3:
                continue

            # Keep at least £3.9m for each unfilled slot.
            remaining_slots = 15 - (len(selected) + 1)
            minimum_future_budget = remaining_slots * 3.9
            if spent + price + minimum_future_budget > budget:
                continue

            selected.append(player)
            spent += price
            club_counts[club] = club_counts.get(club, 0) + 1

    squad = pd.DataFrame(selected)

    # Repair/fill any remaining slots using value and projected points.
    remaining = pool[
        ~pool["id"].isin(
            squad["id"] if not squad.empty else []
        )
    ].sort_values(
        ["value_score", "selection_score"],
        ascending=False,
    )

    for _, player in remaining.iterrows():
        if len(squad) >= 15:
            break

        price = float(player["price"])
        club = int(player["team"])
        position = player["position"]

        if spent + price > budget:
            continue

        pos_count = (
            int((squad["position"] == position).sum())
            if not squad.empty
            else 0
        )

        if pos_count >= limits.get(position, 0):
            continue
        if club_counts.get(club, 0) >= 3:
            continue

        squad = pd.concat(
            [squad, pd.DataFrame([player])],
            ignore_index=True,
        )
        spent += price
        club_counts[club] = club_counts.get(club, 0) + 1

    return squad.reset_index(drop=True)


def best_starting_xi(squad: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Return the highest-scoring legal XI using the fixed 3-4-3 formation."""
    defenders, midfielders, forwards = 3, 4, 3

    gks = squad[squad["position"] == "Goalkeeper"].sort_values(
        "gw_projection", ascending=False
    ).head(1)
    defs = squad[squad["position"] == "Defender"].sort_values(
        "gw_projection", ascending=False
    ).head(defenders)
    mids = squad[squad["position"] == "Midfielder"].sort_values(
        "gw_projection", ascending=False
    ).head(midfielders)
    fwds = squad[squad["position"] == "Forward"].sort_values(
        "gw_projection", ascending=False
    ).head(forwards)

    if len(gks) != 1 or len(defs) != 3 or len(mids) != 4 or len(fwds) != 3:
        return pd.DataFrame(), ""

    xi = pd.concat([gks, defs, mids, fwds], ignore_index=True)
    return xi, "3-4-3"

def set_current_gw_projection(
    players: pd.DataFrame,
    projections: pd.DataFrame,
    gw: int,
) -> pd.DataFrame:
    """Attach the selected GW's projection to each player."""
    gw_data = projections[
        projections["gw"] == gw
    ][["id", "expected_points", "fixture_summary", "fixture_count"]].copy()

    gw_data = gw_data.rename(
        columns={
            "expected_points": "gw_projection",
            "fixture_summary": "gw_fixture_summary",
            "fixture_count": "gw_fixture_count",
        }
    )

    out = players.drop(
        columns=[
            c
            for c in [
                "gw_projection",
                "gw_fixture_summary",
                "gw_fixture_count",
            ]
            if c in players.columns
        ],
        errors="ignore",
    ).merge(
        gw_data,
        on="id",
        how="left",
    )

    out["gw_projection"] = out["gw_projection"].fillna(0.0)
    out["gw_fixture_summary"] = out["gw_fixture_summary"].fillna("BLANK")
    out["gw_fixture_count"] = out["gw_fixture_count"].fillna(0).astype(int)
    return out


def captain_choices(
    xi: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """Rank captain candidates using projection, availability and form.

    The captain receives double points, so the extra expected contribution is
    approximately one additional projected Gameweek score.
    """
    if xi.empty:
        raise ValueError("Cannot choose captain from an empty XI.")

    ranked = xi.copy()
    ranked["captain_weighted"] = (
        ranked["gw_projection"] * 1.0
        + ranked["availability"] * 0.03
        + ranked["form_num"] * 0.15
    )

    ranked = ranked.sort_values(
        ["captain_weighted", "gw_projection"],
        ascending=False,
    ).reset_index(drop=True)

    captain = ranked.iloc[0]
    vice = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
    return captain, vice


def bench_order(squad: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """Order the bench: goalkeeper separately, then three reliable outfielders.

    Outfield bench players are ranked using expected points, availability and
    minutes reliability, while preserving the first-sub priority.
    """
    xi_ids = set(xi["id"].tolist())

    bench = squad[
        ~squad["id"].isin(xi_ids)
    ].copy()

    if bench.empty:
        return bench

    gk = bench[
        bench["position"] == "Goalkeeper"
    ].sort_values(
        ["availability", "gw_projection"],
        ascending=False,
    )

    outfield = bench[
        bench["position"] != "Goalkeeper"
    ].copy()

    outfield["bench_score"] = (
        outfield["gw_projection"] * 0.70
        + outfield["availability"] * 0.20
        + outfield["minutes_rate"] * 0.10
    )

    outfield = outfield.sort_values(
        "bench_score",
        ascending=False,
    )

    ordered = pd.concat(
        [gk.head(1), outfield.head(3)],
        ignore_index=True,
    )
    ordered["bench_order"] = range(1, len(ordered) + 1)
    return ordered


def add_fpl_labels(
    players: pd.DataFrame,
    squad: pd.DataFrame,
    xi: pd.DataFrame,
) -> pd.DataFrame:
    out = players.copy()

    squad_ids = set(squad["id"].tolist()) if not squad.empty else set()
    xi_ids = set(xi["id"].tolist()) if not xi.empty else set()

    out["in_squad"] = out["id"].isin(squad_ids)
    out["starter"] = out["id"].isin(xi_ids)

    out["differential_score"] = (
        out["projection_total"]
        / (1 + out["selected_by_percent_num"])
    )

    return out


def player_table(
    df: pd.DataFrame,
    projection_col: str = "projection_total",
) -> pd.DataFrame:
    cols = [
        "web_name",
        "position",
        "team_name",
        "price",
        projection_col,
        "form_num",
        "availability",
        "selected_by_percent_num",
        "price_trend",
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
        "Price trend",
    ]
    return view


def strategy_player_name(row: pd.Series) -> str:
    return f"{row['web_name']} ({row['team_name']})"


def choose_best_transfer(
    current: pd.Series,
    candidates: pd.DataFrame,
    budget_delta: float,
    current_ids: set,
) -> Optional[pd.Series]:
    """Find the best same-position replacement for one outgoing player."""
    if candidates.empty:
        return None

    position = current["position"]
    allowed = candidates[
        (candidates["position"] == position)
        & (~candidates["id"].isin(current_ids))
    ].copy()

    if allowed.empty:
        return None

    allowed["net_gain"] = (
        allowed["gw_projection"] - float(current["gw_projection"])
    )

    allowed["cost_delta"] = (
        allowed["price"] - float(current["price"])
    )

    allowed = allowed[
        allowed["cost_delta"] <= budget_delta + 1e-9
    ].sort_values(
        ["net_gain", "value_score"],
        ascending=False,
    )

    if allowed.empty:
        return None

    return allowed.iloc[0]


def transfer_strategy(
    starting_squad: pd.DataFrame,
    players: pd.DataFrame,
    projections: pd.DataFrame,
    start_gw: int,
    end_gw: int,
    max_hits_per_gw: int = 1,
) -> pd.DataFrame:
    """Create a rolling transfer plan.

    Starts with the optimized squad for start_gw. For each later GW it:
    - evaluates the next GW projection,
    - banks unused free transfers up to five,
    - identifies the best legal same-position upgrades,
    - only recommends hits when the projected gain exceeds the 4-point cost
      over the selected Gameweek.

    This is a strategy recommendation, not an instruction to execute transfers.
    """
    current = starting_squad.copy()
    rows = []
    free_transfers = 1

    for gw in range(start_gw + 1, end_gw + 1):
        gw_players = set_current_gw_projection(
            players,
            projections,
            gw,
        )

        next_data = gw_players[
            [
                "id",
                "gw_projection",
                "gw_fixture_summary",
                "price_trend",
                "strategy_price",
            ]
        ].rename(
            columns={
                "gw_projection": "next_gw_projection",
                "gw_fixture_summary": "next_gw_fixture_summary",
                "price_trend": "next_price_trend",
                "strategy_price": "next_strategy_price",
            }
        )

        current = current.drop(
            columns=[
                c
                for c in [
                    "gw_projection",
                    "gw_fixture_summary",
                    "price_trend",
                    "strategy_price",
                    "next_gw_projection",
                    "next_gw_fixture_summary",
                    "next_price_trend",
                    "next_strategy_price",
                ]
                if c in current.columns
            ],
            errors="ignore",
        ).merge(
            next_data,
            on="id",
            how="left",
        )

        current["gw_projection"] = current["next_gw_projection"].fillna(0.0)
        current["gw_fixture_summary"] = current[
            "next_gw_fixture_summary"
        ].fillna("BLANK")
        current["price_trend"] = current[
            "next_price_trend"
        ].fillna("Stable")
        current["strategy_price"] = current[
            "next_strategy_price"
        ].fillna(current["price"])

        current_ids = set(current["id"].tolist())
        current_budget = float(current["price"].sum())
        transfer_count = 0

        # A simple greedy transfer loop: repeatedly take the strongest
        # improvement while respecting club/position/budget constraints.
        while transfer_count < max_hits_per_gw:
            candidates = gw_players[
                ~gw_players["id"].isin(current_ids)
            ].copy()

            # Do not recommend unavailable players as upgrades.
            candidates = candidates[
                candidates["availability"] >= 60
            ]

            if candidates.empty:
                break

            best_move = None
            best_gain = 0.0

            for _, outgoing in current.sort_values(
                "gw_projection"
            ).iterrows():
                same_position = candidates[
                    candidates["position"] == outgoing["position"]
                ].copy()

                if same_position.empty:
                    continue

                same_position["gain"] = (
                    same_position["gw_projection"]
                    - float(outgoing["gw_projection"])
                )
                same_position["cost_delta"] = (
                    same_position["price"]
                    - float(outgoing["price"])
                )

                feasible = same_position[
                    same_position["cost_delta"]
                    <= (
                        0.0
                        if transfer_count >= free_transfers
                        else max(
                            0.0,
                            current_budget
                            - current_budget,
                        )
                    )
                ]

                # Recompute affordability relative to the squad's current
                # total. The squad receives the outgoing player's price first.
                feasible = same_position[
                    same_position["price"]
                    <= (
                        current_budget
                        - float(outgoing["price"])
                        + float(outgoing["price"])
                        + 1e-9
                    )
                ]

                for _, incoming in feasible.iterrows():
                    # Temporarily replace outgoing to test the 3-per-club rule.
                    test_clubs = current[
                        current["id"] != outgoing["id"]
                    ]["team"].tolist()

                    if test_clubs.count(int(incoming["team"])) >= 3:
                        continue

                    gain = float(incoming["gw_projection"]) - float(
                        outgoing["gw_projection"]
                    )

                    # If this is a paid transfer, subtract the 4-point hit.
                    hit_cost = (
                        0.0
                        if transfer_count < free_transfers
                        else 4.0
                    )
                    net_gain = gain - hit_cost

                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_move = (
                            outgoing.copy(),
                            incoming.copy(),
                            hit_cost,
                            gain,
                        )

            if best_move is None:
                break

            outgoing, incoming, hit_cost, gross_gain = best_move

            # Only take paid transfers when they clearly beat the hit.
            if transfer_count >= free_transfers and gross_gain <= 4.0:
                break

            current = current[
                current["id"] != outgoing["id"]
            ].copy()

            incoming_row = incoming.copy()
            current = pd.concat(
                [current, pd.DataFrame([incoming_row])],
                ignore_index=True,
            )
            current_ids = set(current["id"].tolist())
            current_budget = float(current["price"].sum())

            transfer_count += 1

            rows.append(
                {
                    "GW": gw,
                    "Transfer": f"{outgoing['web_name']} → {incoming['web_name']}",
                    "Out": outgoing["web_name"],
                    "In": incoming["web_name"],
                    "Position": incoming["position"],
                    "Projected gain": round(gross_gain, 2),
                    "Hit cost": int(hit_cost),
                    "Net gain": round(gross_gain - hit_cost, 2),
                    "Reason": (
                        f"{incoming['gw_projection']:.1f} vs "
                        f"{outgoing['gw_projection']:.1f} projected points; "
                        f"{incoming['price_trend']}"
                    ),
                }
            )

        # Bank a transfer if there was no good move.
        if transfer_count == 0:
            free_transfers = min(5, free_transfers + 1)
            rows.append(
                {
                    "GW": gw,
                    "Transfer": "Roll transfer",
                    "Out": "—",
                    "In": "—",
                    "Position": "—",
                    "Projected gain": 0.0,
                    "Hit cost": 0,
                    "Net gain": 0.0,
                    "Reason": f"No upgrade worth using; banked FT for GW{gw + 1}.",
                }
            )
        else:
            # One free transfer was consumed; additional transfers would have
            # cost points. This model deliberately avoids hits unless worthwhile.
            free_transfers = max(1, free_transfers - transfer_count + 1)
            free_transfers = min(5, free_transfers)

    return pd.DataFrame(rows)


def build_gw_summary(
    players: pd.DataFrame,
    projections: pd.DataFrame,
    start_gw: int,
    end_gw: int,
) -> pd.DataFrame:
    """Summarize each Gameweek's best captain and best XI projection."""
    rows = []

    for gw in range(start_gw, end_gw + 1):
        gw_players = set_current_gw_projection(
            players,
            projections,
            gw,
        )

        # Use the best 15-man squad for that specific GW to assess the
        # Gameweek opportunity, while the transfer planner handles continuity.
        target_pool = gw_players.copy()
        target_pool["projection_total"] = target_pool["gw_projection"]
        target_pool["value_score"] = (
            target_pool["projection_total"]
            / target_pool["price"].replace(0, 99)
        )

        target = greedy_squad(
            target_pool,
            100.0,
            1,
        )

        if target.empty or len(target) < 11:
            continue

        target["gw_projection"] = target["projection_total"]
        xi, formation = best_starting_xi(target)

        if xi.empty:
            continue

        captain, vice = captain_choices(xi)

        rows.append(
            {
                "GW": gw,
                "Formation": formation,
                "XI projection": round(float(xi["gw_projection"].sum()), 1),
                "Captain": captain["web_name"],
                "Captain projection": round(float(captain["gw_projection"]), 1),
                "Vice": vice["web_name"],
                "Best fixture": (
                    xi.sort_values(
                        "gw_projection",
                        ascending=False,
                    ).iloc[0]["gw_fixture_summary"]
                ),
            }
        )

    return pd.DataFrame(rows)



def main():
    st.set_page_config(
        page_title="FPL Scout AI",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ FPL Scout AI")
    st.caption(
        "Gameweek-by-Gameweek FPL optimizer using official FPL data, player history, "
        "fixture difficulty, advanced attacking/defensive signals and optional Google News signals."
    )

    with st.sidebar:
        st.header("⚙️ Strategy settings")

        mode = st.radio(
            "Analysis mode",
            ["Single Gameweek", "Multi-Gameweek"],
        )

        if mode == "Single Gameweek":
            selected_gw = st.number_input(
                "Gameweek",
                min_value=1,
                max_value=38,
                value=1,
                step=1,
            )
            start_gw = int(selected_gw)
            end_gw = int(selected_gw)
        else:
            start_gw = st.number_input(
                "Start Gameweek",
                min_value=1,
                max_value=38,
                value=1,
                step=1,
            )
            end_gw = st.number_input(
                "End Gameweek",
                min_value=int(start_gw),
                max_value=38,
                value=min(int(start_gw) + 5, 38),
                step=1,
            )
            start_gw = int(start_gw)
            end_gw = int(end_gw)

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

        advanced_history = st.checkbox(
            "Use advanced player history model",
            value=True,
            help="Adds home/away history, xG/xA, clean-sheet history, set-piece roles and expected-minutes weighting. Uses extra official FPL API calls.",
        )

        history_limit = st.slider(
            "Deep-history player limit",
            min_value=50,
            max_value=400,
            value=250,
            step=50,
            disabled=not advanced_history,
        )

        ownership_limit = st.slider(
            "Differential ownership ceiling %",
            min_value=1.0,
            max_value=20.0,
            value=8.0,
            step=1.0,
        )

        st.divider()
        st.subheader("Transfer strategy")

        enable_strategy = st.checkbox(
            "Build rolling transfer strategy",
            value=(end_gw - start_gw >= 1),
        )

        max_hits = st.slider(
            "Maximum transfers considered per GW",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            disabled=not enable_strategy,
            help="Extra transfers cost 4 FPL points each. The optimizer only recommends a hit when the projected gain exceeds the cost.",
        )

        st.info(
            "Price-change labels are a market-momentum heuristic. "
            "FPL's exact price-change formula is not public."
        )

    try:
        bootstrap, fixtures = get_fpl_data()
    except Exception as exc:
        st.error(f"Could not reach the official FPL API - {exc}")
        st.stop()

    events = pd.DataFrame(bootstrap.get("events", []))
    if not events.empty:
        unfinished = events[
            events.get("finished", pd.Series(False, index=events.index)).eq(False)
            & events.get("is_previous", pd.Series(False, index=events.index)).eq(False)
        ]
        if not unfinished.empty:
            next_gw = int(unfinished.iloc[0]["id"])
            if start_gw < next_gw:
                st.info(
                    f"The next unfinished FPL Gameweek appears to be GW{next_gw}. "
                    "You can select it in the sidebar."
                )

    with st.spinner(
        "Loading FPL data, calculating fixture-by-fixture projections "
        "and researching the web..."
    ):
        players, fixtures_df = prepare_players(
            bootstrap,
            fixtures,
            with_news=use_news,
            advanced_history=advanced_history,
            history_limit=history_limit,
        )

        players = price_change_signal(players)

        players, projections = apply_fixture_projections(
            players,
            fixtures_df,
            start_gw,
            end_gw,
        )

        if mode == "Single Gameweek":
            players = set_current_gw_projection(
                players,
                projections,
                start_gw,
            )
            players["ranking_projection"] = players["gw_projection"]
        else:
            # For a multi-GW squad, optimize for total expected points across
            # the selected horizon.
            players["ranking_projection"] = players["projection_total"]
            players["gw_projection"] = players["projection_avg"]
            players["gw_fixture_summary"] = players.apply(
                lambda r: f"{int(r['fixture_count'])} fixture(s) across GW{start_gw}–GW{end_gw}",
                axis=1,
            )

        players["value_score"] = (
            players["ranking_projection"]
            / players["price"].replace(0, 99)
        )

        # The squad optimizer expects projection_total, so make it the selected
        # horizon's ranking metric.
        players["projection_total"] = players["ranking_projection"]

        squad = greedy_squad(
            players,
            budget,
            end_gw - start_gw + 1,
        )

        if squad.empty or len(squad) < 15:
            st.error(
                "The optimizer could not build a legal 15-player squad within "
                "the selected budget. Try increasing the budget."
            )
            st.stop()

        # For the XI, always use the selected Gameweek's actual projection.
        squad_ids = set(squad["id"].tolist())
        squad_full = players[
            players["id"].isin(squad_ids)
        ].copy()

        squad_full = set_current_gw_projection(
            squad_full,
            projections,
            start_gw,
        )

        xi, formation = best_starting_xi(squad_full)

        if xi.empty:
            st.error(
                "Could not create a legal starting XI from the optimized squad."
            )
            st.stop()

        captain, vice = captain_choices(xi)
        bench = bench_order(squad_full, xi)

        players = add_fpl_labels(
            players,
            squad_full,
            xi,
        )

        differentials = players[
            (players["selected_by_percent_num"] <= ownership_limit)
            & (~players["in_squad"])
        ].sort_values(
            ["differential_score", "ranking_projection"],
            ascending=False,
        )

        budget_options = players[
            (players["price"] <= 6.0)
            & (~players["in_squad"])
        ].sort_values(
            ["value_score", "ranking_projection"],
            ascending=False,
        )

        avoid = players[
            (~players["in_squad"])
            & (players["price"] >= 4.5)
        ].sort_values(
            ["availability", "ranking_projection"],
            ascending=[True, True],
        )

        if enable_strategy and end_gw > start_gw:
            strategy = transfer_strategy(
                squad_full,
                players,
                projections,
                start_gw,
                end_gw,
                max_hits_per_gw=max_hits,
            )
        else:
            strategy = pd.DataFrame()

        gw_summary = build_gw_summary(
            players,
            projections,
            start_gw,
            end_gw,
        )

    # Header metrics.
    captain_col, vice_col, formation_col, budget_col = st.columns(4)

    with captain_col:
        st.metric(
            "🎯 Captain",
            captain["web_name"],
            f"{captain['gw_projection']:.1f} GW points",
        )

    with vice_col:
        st.metric(
            "🥈 Vice Captain",
            vice["web_name"],
            f"{vice['gw_projection']:.1f} GW points",
        )

    with formation_col:
        st.metric(
            "📐 Best formation",
            formation,
            f"{xi['gw_projection'].sum():.1f} XI projection",
        )

    with budget_col:
        spend = float(squad_full["price"].sum())
        st.metric(
            "💰 Squad spend",
            f"£{spend:.1f}m",
            f"£{budget - spend:.1f}m remaining",
        )

    tabs = st.tabs(
        [
            "🏆 Recommended Team",
            "📅 Gameweek Projections",
            "🔄 Transfer Strategy",
            "💎 Targets & Risks",
            "📰 Research",
        ]
    )

    with tabs[0]:
        st.subheader(
            f"GW{start_gw} starting XI — {formation}"
        )

        show_xi = xi[
            [
                "web_name",
                "position",
                "team_name",
                "price",
                "gw_projection",
                "gw_fixture_summary",
            ]
        ].copy()

        show_xi.columns = [
            "Player",
            "Position",
            "Club",
            "Price",
            "GW Projection",
            "Fixture",
        ]

        st.dataframe(
            show_xi,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("🪑 Bench order")

        if not bench.empty:
            bench_view = bench[
                [
                    "bench_order",
                    "web_name",
                    "position",
                    "team_name",
                    "price",
                    "gw_projection",
                    "availability",
                ]
            ].copy()

            bench_view.columns = [
                "Bench",
                "Player",
                "Position",
                "Club",
                "Price",
                "GW Projection",
                "Availability %",
            ]

            st.dataframe(
                bench_view,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("📋 Full 15-player squad")

        squad_view = squad_full[
            [
                "web_name",
                "position",
                "team_name",
                "price",
                "ranking_projection",
                "gw_projection",
                "price_trend",
            ]
        ].copy()

        squad_view.columns = [
            "Player",
            "Position",
            "Club",
            "Price",
            "Horizon Projection",
            "GW Projection",
            "Price Trend",
        ]

        st.dataframe(
            squad_view,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            f"Captain recommendation: **{captain['web_name']}**. "
            f"The captain is selected using GW projection plus availability/form "
            f"weighting; FPL doubles the captain's Gameweek score."
        )

    with tabs[1]:
        st.subheader("📅 Fixture-by-fixture expected points")

        if not projections.empty:
            selected_ids = set(squad_full["id"].tolist())
            p = projections[
                projections["id"].isin(selected_ids)
            ].copy()

            p = p.merge(
                players[
                    [
                        "id",
                        "web_name",
                        "position",
                        "team_name",
                    ]
                ],
                on="id",
                how="left",
            )

            matrix = p.pivot_table(
                index=["web_name", "position", "team_name"],
                columns="gw",
                values="expected_points",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()

            matrix.columns = [
                (
                    f"GW{c}"
                    if isinstance(c, int)
                    else c
                )
                for c in matrix.columns
            ]

            st.dataframe(
                matrix,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Gameweek captain plan")

        if not gw_summary.empty:
            st.dataframe(
                gw_summary,
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        st.subheader(
            f"🔄 Rolling transfer strategy: GW{start_gw}–GW{end_gw}"
        )

        st.caption(
            "The plan starts from the optimized opening squad, then looks for "
            "same-position upgrades while respecting club limits and transfer costs."
        )

        if strategy.empty:
            st.info(
                "No transfer strategy was generated. Select a range covering at "
                "least two Gameweeks."
            )
        else:
            st.dataframe(
                strategy,
                use_container_width=True,
                hide_index=True,
            )

            hits = int(strategy["Hit cost"].sum())
            gains = float(strategy["Projected gain"].sum())

            a, b, c = st.columns(3)
            with a:
                st.metric("Recommended moves", len(strategy))
            with b:
                st.metric("Projected gross gain", f"{gains:.1f}")
            with c:
                st.metric("Projected hit cost", f"{hits}")

        st.subheader("📌 Strategy principles")
        st.write(
            "- The starting XI is always locked to 3-4-3."
        )
        st.write(
            "- Bank a free transfer when there is no meaningful upgrade."
        )
        st.write(
            "- A paid transfer is only recommended when the projected one-GW gain "
            "exceeds the four-point hit."
        )
        st.write(
            "- Transfers are position-for-position and respect the three-player "
            "club limit."
        )

    with tabs[3]:
        st.subheader("🔥 Best differentials")

        if differentials.empty:
            st.info("No strong low-ownership differential was found.")
        else:
            st.dataframe(
                player_table(
                    differentials.head(10),
                    projection_col="ranking_projection",
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("💰 Best budget options")

        if budget_options.empty:
            st.info("No additional budget options found.")
        else:
            st.dataframe(
                player_table(
                    budget_options.head(10),
                    projection_col="ranking_projection",
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("📈 Price-change / transfer momentum")

        price_watch = players.sort_values(
            ["transfer_momentum", "ranking_projection"],
            ascending=False,
        ).head(15)

        price_view = price_watch[
            [
                "web_name",
                "team_name",
                "price",
                "cost_change_event",
                "net_transfer_event",
                "transfer_momentum",
                "price_trend",
            ]
        ].copy()

        price_view.columns = [
            "Player",
            "Club",
            "Price",
            "GW Price Change",
            "Net Transfers",
            "Momentum",
            "Signal",
        ]

        st.dataframe(
            price_view,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Price-change signal is heuristic. The official FPL price-change "
            "formula is variable and not disclosed publicly."
        )

        st.subheader("⚠️ Players to avoid")

        if avoid.empty:
            st.info("No obvious avoid candidates found.")
        else:
            st.dataframe(
                player_table(
                    avoid.head(10),
                    projection_col="ranking_projection",
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[4]:
        st.subheader("📰 Research notes")

        selected_names = set(xi["web_name"].tolist())

        for _, row in players[
            players["web_name"].isin(selected_names)
        ].iterrows():
            with st.expander(
                f"{row['web_name']} — {row['team_name']}"
            ):
                st.write(
                    f"**GW{start_gw} projection:** "
                    f"{row['gw_projection']:.1f} | "
                    f"**Availability:** {row['availability']:.0f}% | "
                    f"**Price:** £{row['price']:.1f}m | "
                    f"**Price signal:** {row['price_trend']}"
                )

                st.write(
                    f"**Fixture:** {row['gw_fixture_summary']}"
                )

                st.write(
                    f"**Expected minutes:** {row['expected_minutes']:.0f} | "
                    f"**Set-piece score:** {row['set_piece_score']:.2f} | "
                    f"**Home PPG:** {row['home_ppg']:.2f} | "
                    f"**Away PPG:** {row['away_ppg']:.2f}"
                )
                st.write(
                    f"**xG/90:** {row['xg90_hist']:.2f} | "
                    f"**xA/90:** {row['xa90_hist']:.2f} | "
                    f"**Historical clean-sheet rate:** {row['clean_sheet_rate_hist']:.0%}"
                )

                if row["news_headlines"]:
                    st.write(row["news_headlines"])
                else:
                    st.write("No recent headlines returned.")

    st.caption(
        f"Analysis: "
        f"{'GW' + str(start_gw) if start_gw == end_gw else f'GW{start_gw}–GW{end_gw}'}. "
        f"Data refreshed at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        "This is an analytical aid, not financial or betting advice."
    )


if __name__ == "__main__":
    main()

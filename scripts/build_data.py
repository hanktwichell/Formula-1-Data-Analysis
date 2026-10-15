#!/usr/bin/env python3
"""Build the JSON data behind the site from the F1DB SQLite database.

Pipeline
--------
1. Pull every qualifying result since 1950 (Indianapolis 500 excluded — it had
   its own entry list and almost no overlap with the rest of the grid).
2. Pair each driver with their true teammate(s): same entrant *and* same
   constructor at the same round. (Grouping on constructor alone would make
   privateer customer cars "teammates" of the works team.)
3. For each teammate pair at each race, take the deepest qualifying session
   both drivers set a time in (Q3 > Q2 > Q1 in the knockout era, the overall
   best time before that) and record the log-ratio gap in percent.
4. Normalise those gaps by how spread out teammate gaps were that season, so a
   2% gap in 1955 and a 0.3% gap in 2015 are measured on the same scale.
5. Solve the whole teammate web at once: every driver-season gets a rating
   such that the difference between any two teammates' ratings matches their
   observed gaps (robust weighted least squares). Because the system is solved
   jointly, beating a teammate is only worth as much as that teammate's own
   standing elsewhere in the web: Pizzonia is rated through Montoya, Yoong
   through Alonso, and so on. A shared age curve and a light per-season anchor
   keep 75 years of chaining from drifting, and each driver's rating follows a
   penalised random walk across a career. There are no bonuses: every season
   rating is the pace the driver actually showed, rookie years included.
6. Only after the network is solved is uncertainty handled: each driver's
   peak/career estimate is shrunk toward the field in proportion to its
   standard error (empirical Bayes), so thinly-linked drivers can't top the
   list on a lucky chain. No shrinkage is applied *inside* the fit, because
   pulling weak drivers toward average would inflate everyone who beat them.

Run:  python3 scripts/build_data.py            (downloads latest F1DB)
      python3 scripts/build_data.py --db path  (use a local f1db.db)
"""

import argparse
import io
import json
import math
import os
import sqlite3
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache", "f1db")
OUT = os.path.join(ROOT, "site", "data")
F1DB_URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-sqlite.zip"

# ---- model parameters -------------------------------------------------------
MAX_Z = 6.0            # drop race gaps beyond this many season-scales (failures, crashes)
HUBER_K = 2.5          # Huber threshold (in season-scale units) for IRLS
# Races within one teammate pairing share a pair-season error (setup preference,
# number-one-driver treatment, upgrade timing). PAIR_RHO = that variance / race
# variance, so a pairing of n races counts as n / (1 + n * PAIR_RHO) independent
# races and never more than 1 / PAIR_RHO. It's a stated prior, not fitted: the
# driver-season terms absorb pair means, so an in-model estimate is biased to ~0.
PAIR_RHO = 1 / 12
RIDGE = 0.02           # numerical only: weak drivers must be placed by the web, not pulled to average
SEASON_ANCHOR = float(os.environ.get("SEASON_ANCHOR", 20.0))  # light pull of each season's race-weighted average driver toward a common level
AGE_SMOOTH = float(os.environ.get("AGE_SMOOTH", 30.0))  # >0 enables a shared age curve (2nd-difference penalty)
AGE_MIN, AGE_MAX = 20, 42
RIDGE_DRIFT = 0.5      # light pull of each season's drift toward the driver's career level
RANDOM_WALK = 20.0     # how strongly a driver's rating resists changing from one season to the next
PEAK_SEASONS = 5       # peak rating = best run of N consecutive seasons
PEAK_MIN_RACES = 6     # seasons with fewer teammate comparisons can't count toward the peak
MIN_RACES = 10         # distinct races with a valid teammate comparison needed to be ranked
REF_LAP_SECONDS = 90.0 # for converting % to seconds in the UI


def log(*a):
    print(*a, file=sys.stderr)


def fetch_db(path):
    if path:
        return path
    db = os.path.join(CACHE, "f1db.db")
    if os.environ.get("F1DB_OFFLINE") and os.path.exists(db):
        return db
    os.makedirs(CACHE, exist_ok=True)
    log("Downloading", F1DB_URL)
    with urllib.request.urlopen(F1DB_URL) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("f1db.db") as src, open(db, "wb") as dst:
            dst.write(src.read())
    return db


def q(con, sql, *args):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(sql, args)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    args = ap.parse_args()
    con = sqlite3.connect(fetch_db(args.db))

    drivers = {r["id"]: r for r in q(con, "SELECT * FROM driver")}
    countries = {r["id"]: r for r in q(con, "SELECT id, name, demonym, alpha2_code FROM country")}
    constructors = {r["id"]: r for r in q(con, "SELECT id, name, full_name FROM constructor")}
    engines = {r["id"]: r["name"] for r in q(con, "SELECT id, name FROM engine_manufacturer")}
    entrants = {r["id"]: r["name"] for r in q(con, "SELECT id, name FROM entrant")}
    gps = {r["id"]: r for r in q(con, "SELECT id, name, short_name, abbreviation FROM grand_prix")}
    races = {r["id"]: r for r in q(con, "SELECT id, year, round, date, grand_prix_id, qualifying_format, circuit_id FROM race")}
    meta = q(con, "SELECT MAX(r.year) y FROM race r JOIN qualifying_result x ON x.race_id = r.id")[0]

    # (year, round, driver, constructor) -> entrant
    entrant_of = {}
    for r in q(con, "SELECT * FROM season_entrant_driver"):
        for rnd in (r["rounds"] or "").split(";"):
            if rnd.strip():
                entrant_of[(r["year"], int(rnd), r["driver_id"], r["constructor_id"])] = r["entrant_id"]

    # ---- race entries (for team history, including races without a quali time)
    results = q(con, """
        SELECT rr.race_id, rr.driver_id, rr.constructor_id, rr.engine_manufacturer_id,
               rr.position_number, rr.position_text, rr.points, rr.grid_position_number
        FROM race_result rr""")
    quali = q(con, """
        SELECT race_id, driver_id, constructor_id, engine_manufacturer_id, position_number, position_text,
               time_millis, q1_millis, q2_millis, q3_millis
        FROM qualifying_result""")

    def team_key(race_id, driver_id, constructor_id):
        rc = races[race_id]
        ent = entrant_of.get((rc["year"], rc["round"], driver_id, constructor_id))
        return (ent or "?" + constructor_id, constructor_id)

    # ---- teammate comparisons -------------------------------------------------
    by_team = defaultdict(list)
    for r in quali:
        rc = races[r["race_id"]]
        if rc["grand_prix_id"] == "indianapolis":
            continue
        by_team[(r["race_id"],) + team_key(r["race_id"], r["driver_id"], r["constructor_id"])].append(r)

    def session_times(r, fmt):
        if r["q1_millis"] or r["q2_millis"] or r["q3_millis"]:
            return [r["q1_millis"], r["q2_millis"], r["q3_millis"]]
        return [r["time_millis"], None, None]

    comps = []  # one row per (race, driver a, driver b) with a < b
    calib = []  # (Q1, Q2, Q3 gap) for pairs where both drivers reached Q3
    for (race_id, ent, cons), rows in by_team.items():
        seen = {}
        for r in rows:  # a driver can appear twice only in data glitches; keep first
            seen.setdefault(r["driver_id"], r)
        rows = list(seen.values())
        if len(rows) < 2:
            continue
        rc = races[race_id]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = sorted((rows[i], rows[j]), key=lambda x: x["driver_id"])
                ta, tb = session_times(a, rc["qualifying_format"]), session_times(b, rc["qualifying_format"])
                sess = None
                for s in (2, 1, 0):
                    if ta[s] and tb[s]:
                        sess = s
                        break
                if sess is None:
                    continue
                gap = 100.0 * math.log(ta[sess] / tb[sess])  # >0: a slower than b
                if all(ta) and all(tb):
                    calib.append([100.0 * math.log(ta[k_] / tb[k_]) for k_ in range(3)])
                # in the knockout era, if either driver went on to a later session
                # they may not have needed a full lap in this one
                advanced = sess < 2 and bool(ta[sess + 1] or tb[sess + 1])
                comps.append(dict(advanced=advanced, sess=sess,
                    race_id=race_id, year=rc["year"], round=rc["round"], date=rc["date"],
                    a=a["driver_id"], b=b["driver_id"], entrant=ent, constructor=cons,
                    ta=ta[sess], tb=tb[sess], session=("Q%d" % (sess + 1)) if (a["q1_millis"] and b["q1_millis"]) else "Q",
                    gap=gap, pa=a["position_number"], pb=b["position_number"],
                ))
    log("raw teammate comparisons:", len(comps))

    # Knockout-session correction. Drivers who are comfortably through rarely do
    # a full lap in Q1/Q2, so gaps measured there are compressed. Calibrate on
    # pairs who both reached Q3: orient each pair by an independent session and
    # compare the mean gap in Q1 (or Q2) with the same pair's Q3 gap. Then, when
    # a comparison had to fall back to Q1/Q2 because one driver advanced and the
    # other didn't, scale it back up. Pairs eliminated together were both flat
    # out, so they're left alone. The displayed gaps stay the real lap times.
    cal = np.array(calib)
    o2 = np.sign(cal[:, 1]); o1 = np.sign(cal[:, 0])
    ko_factor = {
        0: float(np.mean(o2 * cal[:, 0]) / np.mean(o2 * cal[:, 2])),
        1: float(np.mean(o1 * cal[:, 1]) / np.mean(o1 * cal[:, 2])),
    }
    log("knockout compression (from %d both-in-Q3 pairs): Q1 gaps are %.2fx, Q2 gaps %.2fx the Q3 gap" % (
        len(cal), ko_factor[0], ko_factor[1]))
    n_adj = 0
    for c in comps:
        c["mgap"] = c["gap"]
        if c["advanced"]:
            c["mgap"] = c["gap"] / ko_factor[c["sess"]]
            n_adj += 1
    log("comparisons corrected for an advancing driver:", n_adj)

    # ---- season scale (how spread teammate gaps are in a given era) ---------
    by_year = defaultdict(list)
    for c in comps:
        by_year[c["year"]].append(abs(c["mgap"]))
    years = sorted(by_year)
    raw_scale = {y: float(np.median(by_year[y])) for y in years}
    scale = {}
    for y in years:  # smooth over +-2 seasons, weighted by sample size
        num = den = 0.0
        for yy in range(y - 2, y + 3):
            if yy in by_year:
                w = len(by_year[yy]) * (1.0 if yy == y else 0.5)
                num += w * raw_scale[yy]
                den += w
        scale[y] = num / den
    ref_scale = float(np.median([scale[y] for y in years if 2010 <= y <= meta["y"]]))
    log("reference scale (median |gap|, 2010+): %.3f%%" % ref_scale)
    for y in years[::8]:
        log("  scale %d: %.3f%%" % (y, scale[y]))

    for c in comps:
        c["z"] = c["mgap"] / scale[c["year"]]
        c["valid"] = abs(c["z"]) <= MAX_Z
    valid = [c for c in comps if c["valid"]]
    log("valid comparisons:", len(valid), "dropped:", len(comps) - len(valid))

    # ---- fit -------------------------------------------------------------------
    # rating(driver, season) = base(driver) + drift(driver, season)
    # drift follows a penalised random walk across a driver's seasons, so a career
    # can rise and fall without one shared age curve forcing the shape.
    ids = sorted({c["a"] for c in valid} | {c["b"] for c in valid})
    idx = {d: i for i, d in enumerate(ids)}
    nD = len(ids)
    dseasons = sorted({(c["a"], c["year"]) for c in valid} | {(c["b"], c["year"]) for c in valid})
    sidx = {k: nD + i for i, k in enumerate(dseasons)}
    nS = len(dseasons)
    nA = AGE_MAX - AGE_MIN + 1
    A0 = nD + nS
    nP = A0 + nA
    use_age = AGE_SMOOTH > 0

    def age_col(d, yr):
        b = drivers[d]["date_of_birth"]
        a = yr - int(b[:4]) - (1 if b[5:] > "07-01" else 0)
        return A0 + min(max(a, AGE_MIN), AGE_MAX) - AGE_MIN

    pair_season_n = defaultdict(int)
    for c in valid:
        pair_season_n[(c["a"], c["b"], c["year"])] += 1
    ps_key = [(c["a"], c["b"], c["year"]) for c in valid]
    ps_n = np.array([pair_season_n[k_] for k_ in ps_key], dtype=float)

    def pair_weights(rho):
        # races within one pairing share a pair-season error (setup preference,
        # team focus, upgrade timing), so n races are worth n / (1 + n*rho)
        # independent ones: one teammate-season can never be near-certain proof
        return 1.0 / (1.0 + ps_n * rho)

    base_w = pair_weights(PAIR_RHO)

    rows, cols, vals = [], [], []
    for k, c in enumerate(valid):
        rows += [k, k, k, k]
        cols += [idx[c["a"]], sidx[(c["a"], c["year"])], idx[c["b"]], sidx[(c["b"], c["year"])]]
        vals += [1.0, 1.0, -1.0, -1.0]
        if use_age:
            rows += [k, k]; cols += [age_col(c["a"], c["year"]), age_col(c["b"], c["year"])]; vals += [1.0, -1.0]
    nObs = len(valid)
    X = coo_matrix((vals, (rows, cols)), shape=(nObs, nP)).tocsr()
    y = np.array([c["z"] for c in valid])

    pr, pc, pv = [], [], []
    k = 0
    for i in range(nD):
        pr.append(k); pc.append(i); pv.append(math.sqrt(RIDGE)); k += 1
    for key in dseasons:
        pr.append(k); pc.append(sidx[key]); pv.append(math.sqrt(RIDGE_DRIFT)); k += 1
    seasons_by_driver = defaultdict(list)
    for d, yr in dseasons:
        seasons_by_driver[d].append(yr)
    for d, yrs in seasons_by_driver.items():
        for y0, y1 in zip(yrs, yrs[1:]):
            s_ = math.sqrt(RANDOM_WALK / (y1 - y0))
            pr += [k, k]; pc += [sidx[(d, y0)], sidx[(d, y1)]]; pv += [s_, -s_]; k += 1
    if use_age:
        sm = math.sqrt(AGE_SMOOTH)
        for a in range(1, nA - 1):
            pr += [k, k, k]; pc += [A0 + a - 1, A0 + a, A0 + a + 1]; pv += [sm, -2 * sm, sm]; k += 1
        for a in range(27 - AGE_MIN, 31 - AGE_MIN):
            pr.append(k); pc.append(A0 + a); pv.append(30.0)
        k += 1
    else:
        for a in range(nA):
            pr.append(k); pc.append(A0 + a); pv.append(100.0); k += 1
    # season anchor: without it, slow career improvement chains generation to
    # generation and the whole modern grid drifts faster than the 1970s grid
    season_w = defaultdict(lambda: defaultdict(float))
    for c in valid:
        season_w[c["year"]][c["a"]] += 1.0
        season_w[c["year"]][c["b"]] += 1.0
    sa = math.sqrt(SEASON_ANCHOR)
    for yr, wd in (sorted(season_w.items()) if SEASON_ANCHOR > 0 else []):
        tot = sum(wd.values())
        for d, wt in wd.items():
            pr += [k, k]; pc += [idx[d], sidx[(d, yr)]]; pv += [sa * wt / tot, sa * wt / tot]
        k += 1
    P = coo_matrix((pv, (pr, pc)), shape=(k, nP)).tocsr()
    PtP = (P.T @ P).tocsc()

    def irls(base_w):
        w = base_w.copy()
        for it in range(30):
            A = (X.T @ diags(w) @ X + PtP).tocsc()
            beta = spsolve(A, X.T @ (w * y))
            resid = y - X @ beta
            hub = np.where(np.abs(resid) <= HUBER_K, 1.0, HUBER_K / np.maximum(np.abs(resid), 1e-9))
            w_new = base_w * hub
            if np.max(np.abs(w_new - w)) < 1e-4:
                break
            w = w_new
        return A, beta, resid, w, it + 1

    A, beta, resid, w, its = irls(base_w)
    log("IRLS iterations:", its, "params:", nP)

    sigma2 = float(np.sum(w * resid ** 2) / max(1.0, np.sum(w) - nD))
    cov = np.linalg.inv(A.toarray()) * sigma2

    adj = coo_matrix((np.ones(len(valid)), ([idx[c["a"]] for c in valid], [idx[c["b"]] for c in valid])), shape=(nD, nD))
    ncomp, labels = connected_components(adj, directed=False)
    main_label = np.bincount(labels).argmax()
    log("network components:", ncomp, "main size:", int(np.sum(labels == main_label)), "of", nD)

    # ---- per-driver aggregates ------------------------------------------------
    n_comp = defaultdict(int)
    races_cmp = defaultdict(set)
    season_races = defaultdict(set)
    h2h_w = defaultdict(int)
    teammates_of = defaultdict(set)
    for c in valid:
        for d in (c["a"], c["b"]):
            n_comp[d] += 1
            races_cmp[d].add(c["race_id"])
            season_races[(d, c["year"])].add(c["race_id"])
        teammates_of[c["a"]].add(c["b"]); teammates_of[c["b"]].add(c["a"])
        if c["gap"] < 0:
            h2h_w[c["a"]] += 1
        elif c["gap"] > 0:
            h2h_w[c["b"]] += 1

    def combo(d, yrs, weights=None):
        """Mean of season ratings (z) and its standard error."""
        g = np.zeros(nP)
        weights = weights or [1.0] * len(yrs)
        tot = float(sum(weights))
        for yr, wt in zip(yrs, weights):
            g[idx[d]] += wt / tot
            g[sidx[(d, yr)]] += wt / tot
            if use_age:
                g[age_col(d, yr)] += wt / tot
        nz = np.nonzero(g)[0]
        val = float(g[nz] @ beta[nz])
        var = float(g[nz] @ cov[np.ix_(nz, nz)] @ g[nz])
        return val, math.sqrt(max(var, 0.0))

    season_z = {}
    for d, yrs in seasons_by_driver.items():
        for yr in yrs:
            season_z[(d, yr)] = combo(d, [yr])

    # peak = best run of consecutive seasons (race-weighted), so one lucky year
    # against a weak teammate can't carry it; career = all seasons, race-weighted
    peak_z, career_z, peak_years = {}, {}, {}
    for d, yrs in seasons_by_driver.items():
        solid = [yr for yr in yrs if len(season_races[(d, yr)]) >= PEAK_MIN_RACES] or yrs
        n = min(PEAK_SEASONS, len(solid))
        best = None
        for i in range(len(solid) - n + 1):
            win = solid[i:i + n]
            v = combo(d, win, [len(season_races[(d, yr)]) for yr in win])
            if best is None or v[0] < best[0][0]:
                best = (v, win)
        peak_z[d], peak_years[d] = best
        career_z[d] = combo(d, yrs, [len(season_races[(d, yr)]) for yr in yrs])

    eligible = [d for d in ids if len(races_cmp[d]) >= MIN_RACES and labels[idx[d]] == main_label]
    center = float(np.median([career_z[d][0] for d in eligible]))

    def pct(v):
        return ((v[0] - center) * ref_scale, v[1] * ref_scale)

    def eb_shrink(est):
        """Empirical-Bayes shrink {driver: (value, se)}. The prior mean depends on
        how many races a driver has against teammates: drivers whose careers
        ended after a dozen races were, as a group, well short of average, so an
        uncertain estimate is pulled toward what's typical for that sample size,
        not toward the average ranked driver. Precise estimates barely move."""
        x = np.array([math.log(len(races_cmp[d])) for d in eligible])
        vals = np.array([est[d][0] for d in eligible])
        ses = np.array([est[d][1] for d in eligible])
        wts = 1.0 / (ses ** 2 + 0.05 ** 2)
        Xp = np.vstack([np.ones_like(x), x]).T
        coef = np.linalg.solve(Xp.T @ (Xp * wts[:, None]), Xp.T @ (wts * vals))
        res = vals - Xp @ coef
        tau2 = max(float(np.mean(res ** 2) - np.mean(ses ** 2)), 1e-6)
        out = {}
        for d, (v, e) in est.items():
            mu = coef[0] + coef[1] * math.log(max(len(races_cmp[d]), 1))
            k = tau2 / (tau2 + e * e)
            out[d] = (mu + k * (v - mu), math.sqrt(k) * e)
        log("EB prior mean: %+.3f%% at 10 races, %+.3f%% at 50, %+.3f%% at 200; spread sd %.3f%%, mean shrink %.2f" % (
            coef[0] + coef[1] * math.log(10), coef[0] + coef[1] * math.log(50), coef[0] + coef[1] * math.log(200),
            math.sqrt(tau2), float(np.mean(tau2 / (tau2 + ses ** 2)))))
        return out

    if use_age:
        log("age curve (%):", " ".join("%d:%+.2f" % (AGE_MIN + i, beta[A0 + i] * ref_scale) for i in range(nA)))
    dec = defaultdict(list)
    for (d, yr), v in season_z.items():
        dec[yr // 10 * 10] += [pct(v)[0]] * len(season_races[(d, yr)])
    log("race-weighted mean season rating by decade:", " ".join("%ds:%+.2f" % (k_, np.mean(v)) for k_, v in sorted(dec.items())))
    peak_raw = {d: pct(peak_z[d]) for d in ids}
    career_raw = {d: pct(career_z[d]) for d in ids}
    peak_eb = eb_shrink(peak_raw)
    career_pct = eb_shrink(career_raw)
    rating_pct = {d: peak_eb[d][0] for d in ids}       # negative = faster
    rating_se = {d: peak_eb[d][1] for d in ids}

    # strength of opposition: race-weighted mean season rating of teammates faced
    opp_sum, opp_n = defaultdict(float), defaultdict(int)
    for c in valid:
        for me, other in ((c["a"], c["b"]), (c["b"], c["a"])):
            opp_sum[me] += pct(season_z[(other, c["year"])])[0]
            opp_n[me] += 1
    opposition = {d: opp_sum[d] / opp_n[d] for d in opp_n}

    ranked = sorted(eligible, key=lambda d: rating_pct[d])
    rank_of = {d: i + 1 for i, d in enumerate(ranked)}
    career_rank = {d: i + 1 for i, d in enumerate(sorted(eligible, key=lambda d: career_pct[d][0]))}
    for d in ranked[:110]:
        log("%3d %-28s peak %+.3f%% ±%.3f (raw %+.3f) career %+.3f (#%d) opp %+.3f races=%d tm=%d %s" % (
            rank_of[d], drivers[d]["name"], rating_pct[d], rating_se[d], peak_raw[d][0], career_pct[d][0], career_rank[d],
            opposition[d], len(races_cmp[d]), len(teammates_of[d]), peak_years[d]))

    # ---- team history ---------------------------------------------------------
    seasons_of = defaultdict(lambda: defaultdict(lambda: {"races": 0, "points": 0.0, "wins": 0, "podiums": 0, "best": None, "first": "9999"}))
    entered = {}
    for r in results:
        rc = races[r["race_id"]]
        key = team_key(r["race_id"], r["driver_id"], r["constructor_id"])
        k2 = (rc["year"], key[0], key[1], r["engine_manufacturer_id"])
        s = seasons_of[r["driver_id"]][k2]
        if (r["race_id"], r["driver_id"], key) in entered:
            continue
        entered[(r["race_id"], r["driver_id"], key)] = True
        s["races"] += 1
        s["first"] = min(s["first"], rc["date"])
        s["points"] += float(r["points"] or 0)
        if r["position_number"] == 1:
            s["wins"] += 1
        if r["position_number"] and r["position_number"] <= 3:
            s["podiums"] += 1
        if r["position_number"] and (s["best"] is None or r["position_number"] < s["best"]):
            s["best"] = r["position_number"]

    champ = {}
    for r in q(con, "SELECT year, driver_id, position_number FROM season_driver_standing"):
        champ[(r["year"], r["driver_id"])] = r["position_number"]

    def entrant_name(e, cons):
        if e.startswith("?"):
            return constructors[cons]["name"]
        return entrants.get(e, e)

    def country_obj(cid):
        c = countries.get(cid)
        return {"name": c["name"], "demonym": c["demonym"], "code": (c["alpha2_code"] or "").lower()} if c else None

    def gp_label(race_id):
        rc = races[race_id]
        g = gps[rc["grand_prix_id"]]
        return g["short_name"] or g["name"]

    # ---- write output -----------------------------------------------------------
    os.makedirs(os.path.join(OUT, "drivers"), exist_ok=True)

    comps_by_driver = defaultdict(list)
    for c in comps:
        comps_by_driver[c["a"]].append(c)
        comps_by_driver[c["b"]].append(c)

    def summary(did):
        d = drivers[did]
        yrs = sorted({k[0] for k in seasons_of[did]}) or sorted({races[r["race_id"]]["year"] for r in quali if r["driver_id"] == did})
        out = {
            "id": did,
            "name": d["name"],
            "nat": country_obj(d["nationality_country_id"]),
            "first": yrs[0] if yrs else None,
            "last": yrs[-1] if yrs else None,
            "starts": d["total_race_starts"],
            "wins": d["total_race_wins"],
            "poles": d["total_pole_positions"],
            "titles": d["total_championship_wins"],
        }
        if did in idx:
            out["rating"] = round(rating_pct[did], 4)
            out["se"] = round(rating_se[did], 4)
            out["career"] = round(career_pct[did][0], 4)
            out["careerSe"] = round(career_pct[did][1], 4)
            out["peakYears"] = peak_years[did]
            out["opp"] = round(opposition[did], 4)
            out["n"] = n_comp[did]
            out["h2h"] = h2h_w[did]
            out["tm"] = len(teammates_of[did])
            out["connected"] = bool(labels[idx[did]] == main_label)
        if did in rank_of:
            out["rank"] = rank_of[did]
            out["careerRank"] = career_rank[did]
        return out

    index = [summary(d) for d in drivers if seasons_of[d] or d in idx]
    index.sort(key=lambda s: (s.get("rank", 10 ** 6), s["name"]))

    with open(os.path.join(OUT, "drivers.json"), "w") as f:
        json.dump(index, f, separators=(",", ":"), ensure_ascii=False)

    top = []
    for d in ranked:
        s = summary(d)
        # strongest teammate beaten and most frequent teammate for the leaderboard cards
        tms = defaultdict(lambda: [0, 0, []])
        for c in comps_by_driver[d]:
            if not c["valid"]:
                continue
            other = c["b"] if c["a"] == d else c["a"]
            g = c["gap"] if c["a"] == d else -c["gap"]
            t = tms[other]
            t[0] += 1
            t[1] += g < 0
            t[2].append(g)
        best_tm = sorted(tms.items(), key=lambda kv: -kv[1][0])[:3]
        s["keyTeammates"] = [{"id": o, "name": drivers[o]["name"], "n": v[0], "w": v[1], "med": round(float(np.median(v[2])), 3)} for o, v in best_tm]
        s["teams"] = []
        for k2 in sorted(seasons_of[d], key=lambda k: seasons_of[d][k]["first"]):
            nm = constructors[k2[2]]["name"]
            if nm not in s["teams"]:
                s["teams"].append(nm)
        top.append(s)

    model_meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "lastSeason": meta["y"],
        "refScalePct": round(ref_scale, 4),
        "refLapSeconds": REF_LAP_SECONDS,
        "minRaces": MIN_RACES,
        "comparisons": len(valid),
        "drivers": len(ids),
        "eligible": len(ranked),
        "peakSeasons": PEAK_SEASONS,
        "seasonScale": {str(y): round(scale[y], 4) for y in years},
    }
    with open(os.path.join(OUT, "rankings.json"), "w") as f:
        json.dump({"meta": model_meta, "drivers": top}, f, separators=(",", ":"), ensure_ascii=False)

    # the teammate web: every driver with a comparison, every teammate pairing
    pair = defaultdict(lambda: {"raw": [], "adj": [], "years": set()})
    for c in valid:
        e = pair[(c["a"], c["b"])]
        e["raw"].append(c["gap"])
        e["adj"].append(c["z"] * ref_scale)
        e["years"].add(c["year"])
    node_ids = sorted(ids, key=lambda d: (rank_of.get(d, 10 ** 6), drivers[d]["name"]))
    nidx = {d: i for i, d in enumerate(node_ids)}
    web = {
        # [id, name, peak rating, rank or 0, first season, last season, career rating, in main network]
        "nodes": [[d, drivers[d]["name"], round(rating_pct[d], 3), rank_of.get(d, 0),
                   seasons_by_driver[d][0], seasons_by_driver[d][-1], round(career_pct[d][0], 3),
                   1 if labels[idx[d]] == main_label else 0] for d in node_ids],
        # [a, b, races, median gap a-b raw %, median gap a-b era-adjusted %, first year, last year]
        "edges": [[nidx[a], nidx[b], len(e["raw"]), round(float(np.median(e["raw"])), 3),
                   round(float(np.median(e["adj"])), 3), min(e["years"]), max(e["years"])]
                  for (a, b), e in pair.items()],
    }
    with open(os.path.join(OUT, "web.json"), "w") as f:
        json.dump(web, f, separators=(",", ":"), ensure_ascii=False)

    # profiles
    for did in drivers:
        if not (seasons_of[did] or did in idx):
            continue
        d = drivers[did]
        prof = summary(did)
        prof.update({
            "fullName": d["full_name"],
            "dob": d["date_of_birth"],
            "dod": d["date_of_death"],
            "birthplace": d["place_of_birth"],
            "number": d["permanent_number"],
            "podiums": d["total_podiums"],
            "points": float(d["total_points"]),
            "bestChamp": d["best_championship_position"],
        })
        seasons = []
        for (yr, ent, cons, eng), s in sorted(seasons_of[did].items(), key=lambda kv: kv[1]["first"]):
            seasons.append({
                "year": yr, "entrant": entrant_name(ent, cons), "constructor": constructors[cons]["name"],
                "constructorId": cons, "engine": engines.get(eng, eng), "races": s["races"],
                "points": s["points"], "wins": s["wins"], "podiums": s["podiums"], "best": s["best"],
                "champ": champ.get((yr, did)),
            })
        prof["seasons"] = seasons
        prof["pace"] = [
            {"year": yr, "rating": round(pct(season_z[(did, yr)])[0], 4), "se": round(pct(season_z[(did, yr)])[1], 4),
             "races": len(season_races[(did, yr)])}
            for yr in seasons_by_driver.get(did, [])]

        tm = {}
        for c in sorted(comps_by_driver[did], key=lambda c: (c["date"], c["race_id"])):
            me_a = c["a"] == did
            other = c["b"] if me_a else c["a"]
            g = c["gap"] if me_a else -c["gap"]
            t = tm.setdefault(other, {"id": other, "name": drivers[other]["name"], "races": [], "seasons": {}})
            mt, ot = (c["ta"], c["tb"]) if me_a else (c["tb"], c["ta"])
            mp, op = (c["pa"], c["pb"]) if me_a else (c["pb"], c["pa"])
            t["races"].append([c["year"], c["round"], gp_label(c["race_id"]), constructors[c["constructor"]]["name"],
                               c["session"], mt, ot, round(g, 4), mp, op, 1 if c["valid"] else 0])
        tlist = []
        for other, t in tm.items():
            vg = [r[7] for r in t["races"] if r[10]]
            per = defaultdict(list)
            for r in t["races"]:
                if r[10]:
                    per[r[0]].append(r[7])
            t["n"] = len(vg)
            t["w"] = sum(1 for g in vg if g < 0)
            t["l"] = sum(1 for g in vg if g > 0)
            t["median"] = round(float(np.median(vg)), 4) if vg else None
            adj = [r[7] / scale[r[0]] * ref_scale for r in t["races"] if r[10]]
            t["adjMedian"] = round(float(np.median(adj)), 4) if adj else None
            t["mean"] = round(float(np.mean(vg)), 4) if vg else None
            t["years"] = sorted({r[0] for r in t["races"]})
            t["teams"] = sorted({r[3] for r in t["races"]}, key=lambda x: [r[3] for r in t["races"]].index(x))
            t["bySeason"] = [{"year": y, "n": len(v), "w": sum(1 for g in v if g < 0), "median": round(float(np.median(v)), 4)} for y, v in sorted(per.items())]
            if other in idx:
                t["rating"] = round(rating_pct[other], 4)
                t["rank"] = rank_of.get(other)
            # gap implied by the two drivers' season ratings in the years they shared a car
            ex = [(season_z[(did, r[0])][0] - season_z[(other, r[0])][0]) * ref_scale
                  for r in t["races"] if r[10] and (did, r[0]) in season_z and (other, r[0]) in season_z]
            if ex:
                t["expected"] = round(float(np.mean(ex)), 4)
            del t["seasons"]
            tlist.append(t)
        tlist.sort(key=lambda t: (t["years"][0], -t["n"]))
        prof["teammates"] = tlist
        with open(os.path.join(OUT, "drivers", did + ".json"), "w") as f:
            json.dump(prof, f, separators=(",", ":"), ensure_ascii=False)

    log("wrote", len(index), "drivers,", len(ranked), "ranked")


if __name__ == "__main__":
    main()

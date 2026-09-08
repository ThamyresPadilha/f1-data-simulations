import time
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import norm


def parse_lap_time(t_str):
  if pd.isna(t_str) or not isinstance(t_str, str):
    return None
  try:
    parts = t_str.split(":")
    if len(parts) == 2:
      return float(parts[0]) * 60 + float(parts[1])
    return float(t_str)
  except ValueError:
    return None


def init_linear_regressor_matrix(data, num_of_drivers, col_to_rank):
  wins = np.zeros((data.shape[0], num_of_drivers))
  score_diff = np.zeros(data.shape[0])

  for index, row in data.iterrows():
    idx1 = int(row["Driver1"])
    idx2 = int(row["Driver2"])
    if row["Driver1_laptime"] - row["Driver2_laptime"] > 0:
      wins[index][idx1] = -1
      wins[index][idx2] = 1
      score_diff[index] = row["Driver1_laptime"] - row["Driver2_laptime"]
    else:
      wins[index][idx1] = 1
      wins[index][idx2] = -1
      score_diff[index] = row["Driver2_laptime"] - row["Driver1_laptime"]
  wins_df = pd.DataFrame(wins)
  wins_df[col_to_rank] = score_diff
  return wins_df


def massey(data, num_of_drivers, col_to_rank="delta"):
  wins_df = init_linear_regressor_matrix(data, num_of_drivers, col_to_rank)
  model = sm.OLS(wins_df[col_to_rank], wins_df.drop(columns=[col_to_rank]))
  results = model.fit(cov_type="HC1")
  rankings = pd.DataFrame(results.params)
  rankings["std"] = np.sqrt(np.diag(results.cov_params()))
  rankings["consistency"] = (norm.ppf(0.9) - norm.ppf(0.1)) * rankings["std"]
  rankings = (
      rankings.sort_values(by=0, ascending=False)
      .reset_index()
      .rename(columns={"index": "Driver", 0: "massey"})
  )
  rankings = rankings.sort_values(by=["massey"], ascending=False)
  rankings["massey_new"] = rankings["massey"].max() - rankings["massey"]
  return rankings[["Driver", "massey_new"]]


def main():
  anos = list(range(1983, 2025))
  duelos = []

  print("Executando o modelo oficial AWS (Método de Massey) de 1983 a 2025...")

  for ano in anos:
    url = f"https://api.jolpi.ca/ergast/f1/{ano}/qualifying.json?limit=1000"
    try:
      res = requests.get(url, timeout=15)
      if res.status_code != 200:
        continue
      races = res.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
      for race in races:
        results = race.get("QualifyingResults", [])
        if not results:
          continue
        rows = []
        for q in results:
          driver = (
              q.get("Driver", {}).get("driverId", "UNK")
              or q.get("Driver", {}).get("code", "UNK")
          ).upper()
          team = q.get("Constructor", {}).get("constructorId", "UNK")
          tempo_str = q.get("Q3") or q.get("Q2") or q.get("Q1")
          t_sec = parse_lap_time(tempo_str)
          if t_sec and t_sec > 0:
            rows.append({"Piloto": driver, "Equipe": team, "Tempo": t_sec})
        df_race = pd.DataFrame(rows)
        if df_race.empty:
          continue
        for _, grupo in df_race.groupby("Equipe"):
          if len(grupo) == 2:
            p = grupo["Piloto"].tolist()
            t = grupo["Tempo"].tolist()
            diff = t[1] - t[0]
            if abs(diff) <= 0.8:
              duelos.append(
                  {
                      "PilotoA": p[0],
                      "PilotoB": p[1],
                      "TempoA": t[0],
                      "TempoB": t[1],
                  }
              )
    except Exception:
      continue
    time.sleep(0.05)

  df_rede = pd.DataFrame(duelos)
  if df_rede.empty:
    print("Nenhum duelo extraído.")
    return

  todos_pilotos = sorted(
      list(set(df_rede["PilotoA"]).union(set(df_rede["PilotoB"])))
  )
  piloto_to_idx = {p: i for i, p in enumerate(todos_pilotos)}
  idx_to_piloto = {i: p for p, i in piloto_to_idx.items()}

  df_rede["Driver1"] = df_rede["PilotoA"].map(piloto_to_idx)
  df_rede["Driver2"] = df_rede["PilotoB"].map(piloto_to_idx)
  df_rede = df_rede.rename(
      columns={"TempoA": "Driver1_laptime", "TempoB": "Driver2_laptime"}
  )

  counts = pd.concat([df_rede["Driver1"], df_rede["Driver2"]]).value_counts()
  validos = counts[counts >= 8].index
  df_rede = df_rede[
      df_rede["Driver1"].isin(validos) & df_rede["Driver2"].isin(validos)
  ].reset_index(drop=True)

  pilotos_filtrados = sorted(
      list(
          set(df_rede["Driver1"].map(idx_to_piloto)).union(
              set(df_rede["Driver2"].map(idx_to_piloto))
          )
      )
  )
  novo_piloto_to_idx = {p: i for i, p in enumerate(pilotos_filtrados)}
  novo_idx_to_piloto = {i: p for p, i in novo_piloto_to_idx.items()}

  df_rede["Driver1"] = (
      df_rede["Driver1"].map(idx_to_piloto).map(novo_piloto_to_idx)
  )
  df_rede["Driver2"] = (
      df_rede["Driver2"].map(idx_to_piloto).map(novo_piloto_to_idx)
  )

  num_of_drivers = len(pilotos_filtrados)

  ranking_df = massey(df_rede, num_of_drivers)
  ranking_df["Driver"] = ranking_df["Driver"].map(novo_idx_to_piloto)
  ranking_df = ranking_df.rename(
      columns={"Driver": "Piloto", "massey_new": "Atraso (Segundos)"}
  )

  print("\n" + "=" * 55)
  print(" RANKING OFICIAL AWS - MÉTODO DE MASSEY (1983 - 2025)")
  print("=" * 55)
  print(ranking_df.head(20).to_string(index=False))


if __name__ == "__main__":
  main()
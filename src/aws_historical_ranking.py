import fastf1
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
fastf1.Cache.enable_cache('data/cache')

# Escopo completo: de 1983 até 2025
anos = list(range(1983, 2026))
duelos_matriz = []

print("Construindo a teia histórica de confrontos (1983 - 2025)...")
print("Isso pode levar alguns minutos na primeira execução devido ao volume de anos.")

for ano in anos:
    try:
        schedule = fastf1.get_event_schedule(ano)
        for _, event in schedule.iterrows():
            round_num = event['RoundNumber']
            if round_num <= 0:
                continue
                
            try:
                session = fastf1.get_session(ano, round_num, 'Q')
                session.load(telemetry=False, weather=False)
                
                laps = session.laps.pick_accurate()
                if laps.empty:
                    continue
                
                melhores = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
                melhores['LapTimeSeconds'] = melhores['LapTime'].dt.total_seconds()
                
                for equipe, grupo in melhores.groupby('Team'):
                    if len(grupo) == 2:
                        p = grupo['Driver'].tolist()
                        t = grupo['LapTimeSeconds'].tolist()
                        diff = t[1] - t[0]
                        
                        if abs(diff) < 2.0:
                            duelos_matriz.append({
                                'Ano': ano,
                                'GP': event['EventName'],
                                'PilotoA': p[0],
                                'PilotoB': p[1],
                                'Diff': diff
                            })
            except Exception:
                continue
    except Exception:
        continue

df_rede = pd.DataFrame(duelos_matriz)

if df_rede.empty:
    print("Nenhum duelo encontrado no escopo selecionado.")
else:
    print(f"\nTotal histórico de arestas processadas: {len(df_rede)}")
    
    pilotos_unicos = sorted(list(set(df_rede['PilotoA']).union(set(df_rede['PilotoB']))))
    piloto_to_idx = {p: i for i, p in enumerate(pilotos_unicos)}
    
    X = np.zeros((len(df_rede), len(pilotos_unicos)))
    y = np.zeros(len(df_rede))
    
    for idx, row in df_rede.iterrows():
        iA = piloto_to_idx[row['PilotoA']]
        iB = piloto_to_idx[row['PilotoB']]
        X[idx, iA] = -1
        X[idx, iB] = 1
        y[idx] = row['Diff']
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    
    ranking_df = pd.DataFrame({
        'Piloto': pilotos_unicos,
        'DeltaVelocidade (s)': reg.coef_
    })
    
    ranking_df['DeltaVelocidade (s)'] = ranking_df['DeltaVelocidade (s)'] - ranking_df['DeltaVelocidade (s)'].min()
    ranking_df = ranking_df.sort_values(by='DeltaVelocidade (s)', ascending=True)
    
    print("\n=== RANKING HISTÓRICO GLOBAL (1983 - 2025) ===")
    print(ranking_df.head(15).to_string(index=False))
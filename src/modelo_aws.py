import fastf1
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
fastf1.Cache.enable_cache('data/cache')

# Definimos o leque histórico de 2018 até 2026 
# (Você pode estender para anos anteriores conforme o FastF1 baixar os dados)
anos = list(range(2018, 2027))
gps_chave = ['Monaco', 'Great Britain', 'Italy', 'Brazil', 'Belgium', 'Bahrain']

duelos_matriz = []

print("Mapeando a rede global de confrontos entre companheiros...")

for ano in anos:
    for gp in gps_chave:
        try:
            session = fastf1.get_session(ano, gp, 'Q')
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
                    diff = t[1] - t[0] # TempoPilotoB - TempoPilotoA
                    
                    if abs(diff) < 2.0:
                        duelos_matriz.append({
                            'PilotoA': p[0],
                            'PilotoB': p[1],
                            'Diff': diff
                        })
        except Exception:
            pass

df_rede = pd.DataFrame(duelos_matriz)

if df_rede.empty:
    print("Nenhum duelo suficiente encontrado com os filtros atuais.")
else:
    print(f"Total de arestas processadas na rede: {len(df_rede)}")
    
    # Construção da Matriz de Projeto para Regressão Linear (Estilo AWS / Massey)
    pilotos_unicos = sorted(list(set(df_rede['PilotoA']).union(set(df_rede['PilotoB']))))
    piloto_to_idx = {p: i for i, p in enumerate(pilotos_unicos)}
    
    X = np.zeros((len(df_rede), len(pilotos_unicos)))
    y = np.zeros(len(df_rede))
    
    for idx, row in df_rede.iterrows():
        iA = piloto_to_idx[row['PilotoA']]
        iB = piloto_to_idx[row['PilotoB']]
        # Se PilotoA é mais rápido que PilotoB, a diferença t[1] - t[0] é positiva
        X[idx, iA] = -1
        X[idx, iB] = 1
        y[idx] = row['Diff']
    
    # Resolvendo o sistema por mínimos quadrados (define a "habilidade" relativa de cada piloto)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    
    # Criando o ranking final
    ranking_df = pd.DataFrame({
        'Piloto': pilotos_unicos,
        'DeltaVelocidade (s)': reg.coef_
    })
    
    # Normaliza em relação ao piloto mais rápido da amostra (referência = 0.0)
    ranking_df['DeltaVelocidade (s)'] = ranking_df['DeltaVelocidade (s)'] - ranking_df['DeltaVelocidade (s)'].min()
    ranking_df = ranking_df.sort_values(by='DeltaVelocidade (s)', ascending=True)
    
    print("\n=== RANKING DE CLASSIFICAÇÃO ATUALIZADO (ESTILO AWS) ===")
    print(ranking_df.head(15).to_string(index=False))
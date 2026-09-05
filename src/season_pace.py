import fastf1
import pandas as pd
import warnings

# Suprime avisos desnecessários do Pandas no terminal
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

fastf1.Cache.enable_cache('data/cache')

ano = 2026

gps_disputados = [
    'Australia', 'China', 'Japan', 'Miami', 
    'Canada', 'Monaco', 'Spain', 'Austria', 
    'Great Britain', 'Belgium', 'Hungary', 'Netherlands'
]

registros_por_gp = []

print("Processando ritmo normalizado por GP...")

for gp in gps_disputados:
    try:
        session = fastf1.get_session(ano, gp, 'R')
        session.load(telemetry=False, weather=False)
        
        laps = session.laps.pick_accurate().copy()
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        medianas_gp = laps.groupby('Driver')['LapTimeSeconds'].median().reset_index()
        medianas_gp['GrandPrix'] = gp
        
        registros_por_gp.append(medianas_gp)
    except Exception as e:
        print(f"Aviso ao carregar {gp}: {e}")

df_geral = pd.concat(registros_por_gp, ignore_index=True)

season_pace = df_geral.groupby('Driver').agg(
    MediaDasMedianas=('LapTimeSeconds', 'mean'),
    GPsContabilizados=('GrandPrix', 'count')
).reset_index()

# FILTRO DE RELEVÂNCIA: Exige que o piloto tenha participado de pelo menos 5 GPs na amostra
season_pace = season_pace[season_pace['GPsContabilizados'] >= 5]

season_pace = season_pace.sort_values(by='MediaDasMedianas', ascending=True)

print("\n=== RANKING DE RITMO DE CORRIDA (NORMALIZADO E FILTRADO) ===")
print(season_pace.to_string(index=False))
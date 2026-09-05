import fastf1
import pandas as pd

# Habilita o cache
fastf1.Cache.enable_cache('data/cache')

# Carrega a corrida
session = fastf1.get_session(2026, 'Belgium', 'R')
session.load()

# Pega apenas as voltas limpas/precisas
laps = session.laps.pick_accurate()

# Converte o tempo de volta para segundos para facilitar cálculos matemáticos
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Agrupa por piloto e calcula a mediana do tempo de volta e o total de voltas válidas
race_pace = laps.groupby('Driver')['LapTimeSeconds'].agg(['median', 'count']).reset_index()

# Ordena do menor tempo (mais rápido) para o maior
race_pace = race_pace.sort_values(by='median', ascending=True)

# Renomeia as colunas para melhor visualização
race_pace.columns = ['Piloto', 'MedianaLapTime (s)', 'VoltasValidas']

print("=== RANKING DE RITMO DE CORRIDA (RACE PACE) ===")
print(race_pace.head(10).to_string(index=False))


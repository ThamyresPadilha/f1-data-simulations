import fastf1
import pandas as pd

# Habilita o cache local para salvar os dados da API da F1
fastf1.Cache.enable_cache('data/cache')

# Definindo a temporada atual e um GP recente (ex: GP da Bélgica de 2026)
ano = 2026
gp = 'Belgium'
sessao = 'R' # 'R' para Corrida

print(f"Carregando dados da F1 {ano} - {gp}...")
session = fastf1.get_session(ano, gp, sessao)
session.load()

# Extraindo as voltas da corrida
laps = session.laps

# Filtrando apenas voltas limpas/precisas (sem voltas de entrada/saída de box)
clean_laps = laps.pick_accurate()

print(f"\nDados carregados com sucesso!")
print(f"Total de voltas precisas registradas na corrida: {len(clean_laps)}")

# Exibindo as 5 voltas mais rápidas da sessão e seus respectivos pilotos
print("\nAs 5 voltas mais rápidas da corrida:")
print(clean_laps[['Driver', 'LapTime', 'Compound', 'TyreLife']].head(5))

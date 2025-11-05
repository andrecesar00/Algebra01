import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# --- 1. Definir Caminhos ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

file_mandante = BASE_DIR / 'Gols Mandante.csv'
file_visitante = BASE_DIR / 'Gols Visita.csv'

# --- 2. Processar Vetor MANDANTE ---
try:
    print(f"A processar '{file_mandante}'...")
    df_mandante = pd.read_csv(file_mandante)
    if 'gols_mandante' not in df_mandante.columns: sys.exit()
    df_mandante['gols_mandante'] = pd.to_numeric(df_mandante['gols_mandante'], errors='coerce').fillna(0).astype(int)
    vetor_mandante = df_mandante.groupby('ano_campeonato')['gols_mandante'].sum().reset_index()
    vetor_mandante = vetor_mandante[vetor_mandante['ano_campeonato'] >= 2003].copy()
except FileNotFoundError:
    print(f"ERRO: Ficheiro '{file_mandante}' não encontrado.")
    sys.exit()

# --- 3. Processar Vetor VISITANTE ---
try:
    print(f"A processar '{file_visitante}'...")
    df_visitante = pd.read_csv(file_visitante)
    if 'gols_visitante' not in df_visitante.columns: sys.exit()
    df_visitante['gols_visitante'] = pd.to_numeric(df_visitante['gols_visitante'], errors='coerce').fillna(0).astype(int)
    vetor_visitante = df_visitante.groupby('ano_campeonato')['gols_visitante'].sum().reset_index()
    vetor_visitante = vetor_visitante[vetor_visitante['ano_campeonato'] >= 2003].copy()
except FileNotFoundError:
    print(f"ERRO: Ficheiro '{file_visitante}' não encontrado.")
    sys.exit()

# --- 4. Alinhar Vetores e Criar Tabela Comparativa ---
print("\nA alinhar os vetores...")
df_comparativo = pd.merge(vetor_mandante, vetor_visitante, on='ano_campeonato', how='outer')
df_comparativo['gols_mandante'] = df_comparativo['gols_mandante'].fillna(0).astype(int)
df_comparativo['gols_visitante'] = df_comparativo['gols_visitante'].fillna(0).astype(int)
df_comparativo = df_comparativo.sort_values(by='ano_campeonato').reset_index(drop=True)

# --- 5. Adicionar coluna 'mais_gols_em' ---
df_comparativo['mais_gols_em'] = np.where(
    df_comparativo['gols_mandante'] > df_comparativo['gols_visitante'], 
    'Mandante',
    np.where(
        df_comparativo['gols_visitante'] > df_comparativo['gols_mandante'], 
        'Visitante',
        'Empate'
    )
)

# --- 6. Mostrar Análise Ano a Ano ---
print("\n" + "="*50)
print("--- ANÁLISE COMPARATIVA ANO A ANO ---")
print("="*50)
pd.set_option('display.max_rows', None)
print(df_comparativo.to_string(index=False))
pd.reset_option('display.max_rows')

# --- 7. Mostrar Análise de Totais ---
print("\n" + "="*50)
print("--- ANÁLISE DE TOTAIS (2003-2024) ---")
print("="*50)
total_mandante = df_comparativo['gols_mandante'].sum()
total_visitante = df_comparativo['gols_visitante'].sum()
print(f"Total de Gols como Mandante: {total_mandante}")
print(f"Total de GGols como Visitante: {total_visitante}")
if total_mandante > total_visitante:
    print("Resultado: O São Paulo marcou mais gols como Mandante no período.")
elif total_visitante > total_mandante:
    print("Resultado: O São Paulo marcou mais gols como Visitante no período.")
else:
    print("Resultado: O São Paulo marcou o mesmo número de gols como Mandante e Visitante.")

# --- 8. Mostrar Análise Total (Similaridade de Cossenos) ---
print("\n" + "="*50)
print("--- ANÁLISE TOTAL (SIMILARIDADE DE PADRÃO) ---")
print("="*50)
v1_mandante = df_comparativo['gols_mandante'].values
v2_visitante = df_comparativo['gols_visitante'].values
if np.sum(v1_mandante) == 0 or np.sum(v2_visitante) == 0:
    print("ERRO: Vetores nulos.")
    sys.exit()
v1_reshaped = v1_mandante.reshape(1, -1)
v2_reshaped = v2_visitante.reshape(1, -1)
similarity = cosine_similarity(v1_reshaped, v2_reshaped)
score = similarity[0][0]
print(f"Vetor Mandante (Gols/ano): {v1_mandante}")
print(f"Vetor Visitante (Gols/ano): {v2_visitante}")
print("\n" + "-"*50 + f"\nPontuação de Similaridade de Cossenos: {score:.4f}\n" + "-"*50)
print("Explicação: Padrões ALTAMENTE SIMILARES.")

# --- 9. Salvar os ficheiros finais ---
try:
    # Salva o CSV (como antes)
    csv_file = BASE_DIR / 'comparativo_gols_ano_a_ano.csv'
    df_comparativo.to_csv(csv_file, index=False)
    print(f"\nSucesso! Ficheiro CSV guardado em: {csv_file}")
    
    # --- [CORRIGIDO] SALVAR A TABELA EM FORMATO LATEX ---
    latex_file = BASE_DIR / 'tabela_analise.tex'
    # Removido 'booktabs=True' para funcionar em versões mais antigas do pandas
    df_comparativo.to_latex(latex_file, index=False) 
    print(f"Sucesso! Ficheiro LaTeX da tabela guardado em: {latex_file}")
    
except Exception as e:
    print(f"\nErro ao salvar ficheiros de resumo: {e}")
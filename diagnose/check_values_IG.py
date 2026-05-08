import pandas as pd
import json
from pathlib import Path

def main():
    # 1. Carregar configuração e caminhos
    with open('config.local.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    base_dir = Path(config['data_dir'])
    input_dir = base_dir / "inputs" / "clean"
    output_path = base_dir / "outputs" / "estatisticas_indicadores_gestao.xlsx"
    
    # Mapeamento corrigido conforme as colunas reais do Parquet
    # Baseado no log de erro, as colunas seguem o padrão g1_inv_abs, etc.
    indicadores = {
        "g1": {"file": "br_h3_g1_mun_investimento_despesas.parquet", "col": "g1_inv_abs", "tipo": "media"},
        "g2": {"file": "br_h3_g2_mun_planejamento_contingencia.parquet", "col": "g2_pla_norm", "tipo": "binario"},
        "g3": {"file": "br_h3_g3_mun_participacao_nupdec.parquet", "col": "g3_par_norm", "tipo": "binario"},
        "g4": {"file": "br_h3_g4_mun_governanca_conselhos.parquet", "col": "g4_gov_norm", "tipo": "binario"},
        "g5": {"file": "br_h3_g5_mun_resposta_alerta.parquet", "col": "g5_res_norm", "tipo": "binario"},
        "g6": {"file": "br_h3_g6_mun_informacao_mapeamento.parquet", "col": "g6_map_norm", "tipo": "binario"},
        "g7": {"file": "br_h3_g7_mun_reconhecimento_cadastro.parquet", "col": "g7_rec_norm", "tipo": "binario"},
        "g8": {"file": "br_h3_g8_mun_reparacao_direitos.parquet", "col": "g8_rep_abs", "tipo": "media"}
    }
    
    # 2. Carregar Metadados Base (Mapeamento H3 -> Município)
    # Procuramos o arquivo de metadados na pasta data/ (ajuste se o caminho for diferente)
    metadata_path = base_dir / "br_h3_base_metadata.parquet"
    if not metadata_path.exists():
        # Tenta procurar na pasta results caso não esteja na raiz de data
        metadata_path = base_dir / "outputs" / "results" / "br_h3_iic_v2_0_20260427_144058.parquet"
        
    print(f"Lendo metadados de: {metadata_path.name}")
    df_base = pd.read_parquet(metadata_path, columns=['h3_id', 'cd_mun'])
    df_base['cd_mun'] = df_base['cd_mun'].astype(str)

    stats_list = []

    print("Calculando estatísticas nacionais por município...")

    for ref, info in indicadores.items():
        file_path = input_dir / info["file"]
        
        if not file_path.exists():
            print(f"[-] Arquivo não encontrado: {info['file']}")
            continue

        # Lê apenas h3_id e o valor do indicador
        try:
            df_ind = pd.read_parquet(file_path, columns=['h3_id', info["col"]])
            
            # Merge com a base para recuperar o cd_mun
            df = df_ind.merge(df_base, on='h3_id', how='left')
            
            if info["tipo"] == "binario":
                # Agrupa por município e pega o valor (se um hexágono é 1, o município é 1)
                mun_data = df.groupby('cd_mun')[info["col"]].max()
                percentual_1 = (mun_data == 1).mean() * 100
                stats_list.append({
                    "ID": ref.upper(),
                    "Indicador": info["col"],
                    "Métrica": "Percentual de Municípios (Sim/1)",
                    "Valor": f"{percentual_1:.2f}%"
                })
            else:
                # Agrupa por município e tira a média (evita viés de área)
                mun_data = df.groupby('cd_mun')[info["col"]].mean()
                media_nacional = mun_data.mean()
                stats_list.append({
                    "ID": ref.upper(),
                    "Indicador": info["col"],
                    "Métrica": "Média Nacional",
                    "Valor": round(media_nacional, 4)
                })
            print(f"[✓] {ref.upper()} processado.")
        except Exception as e:
            print(f"[!] Erro ao processar {ref.upper()}: {e}")

    # 3. Exportar
    df_resumo = pd.DataFrame(stats_list)
    df_resumo.to_excel(output_path, index=False)
    print(f"\n✅ Relatório gerado: {output_path}")

if __name__ == "__main__":
    main()
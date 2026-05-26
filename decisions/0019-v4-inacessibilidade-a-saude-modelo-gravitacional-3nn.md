# ADR-0019: Adotar modelo gravitacional 3-NN para o indicador v4 (Inacessibilidade à saúde)

## Status

Accepted — 2026-05-19

## Contexto

A dimensão IV precisa de um indicador de acesso a saúde — vulnerabilidade não se reduz a renda ou moradia, e proximidade a equipamentos de saúde afeta a capacidade de responder a impactos climáticos (exposição prolongada a calor, doenças respiratórias agravadas por queimadas, atendimento pós-desastre). O DataSus georreferenciado é restrito (microdados via LAI; endereços em campo livre, sem geocodificação automatizável); CRAS e estabelecimentos de saúde do CNES são as únicas bases abertas georreferenciadas com cobertura nacional. O CNES é a fonte selecionada. A pergunta seguinte: como traduzir pontos de estabelecimentos em um indicador de inacessibilidade por hexágono.

## Decisão

Calcular **v4 — Inacessibilidade à saúde** por um **modelo gravitacional simples sobre os três estabelecimentos mais próximos (3-NN)** de cada hexágono. Para cada hexágono, identificar os três estabelecimentos mais próximos via árvore KD (`scipy.spatial.cKDTree`) com distâncias euclidianas em metros. Calcular a acessibilidade bruta como:

```
v4_abs = Σ_j [ capacity_score_j / (distância_j + 100) ]
```

onde `capacity_score` de cada estabelecimento é a soma de seis categorias binárias de serviço (centro cirúrgico, obstétrico, neonatal, atendimento hospitalar, apoio, ambulatorial) acrescida de 1 (mínimo = 1); o buffer de 100 m no denominador evita divisão por zero. O resultado é normalizado por **min-max com winsorização p3-p97** (exceção mais estreita ao padrão p1-p99 do ADR-0012, para suprimir o efeito de grandes clusters hospitalares como capitais com dezenas de estabelecimentos concentrados). O valor final é **invertido** (`1 − norm`): maior acessibilidade gravitacional bruta → menor inacessibilidade → menor vulnerabilidade.

Direção semântica: maior `v4_norm` = maior inacessibilidade = maior vulnerabilidade.

## Alternativas consideradas

- **2SFCA (Two-Step Floating Catchment Area)**: modelo gravitacional em dois passos que divide a capacidade de cada estabelecimento pela demanda populacional dentro de um raio de catchment antes de somar as contribuições. Corrige um artefato conhecido do modelo simples — cidades pequenas com poucos estabelecimentos concentrados parecem mais acessíveis do que grandes cidades. Implementação preservada em `etl/discarded/v4_cnes_2sfca.py`. **Descartado** porque: (i) sensível a parâmetros arbitrários (raio de catchment), exigindo justificativa adicional e análise de sensibilidade que aumenta a superfície de questionamento em revisão; (ii) custo computacional alto (cálculo de demanda agregada por estabelecimento dentro do raio para todo o Brasil); (iii) o "defeito" do modelo simples é defensável — cidades pequenas com poucos estabelecimentos *de fato* têm acesso geograficamente mais próximo, mesmo que a oferta total seja menor.
- **Distância ao estabelecimento mais próximo (1-NN)**: simples, mas perde a noção de capacidade e ignora estabelecimentos secundários de referência.
- **Modelo gravitacional sem corte (todos os estabelecimentos do país)**: maximiza informação, mas torna o cálculo caro e dilui o sinal local com contribuições negligenciáveis de estabelecimentos distantes.
- **3-NN com modelo gravitacional (escolhido)**: equilibra simplicidade conceitual com captura de capacidade e múltiplos estabelecimentos de referência; computacionalmente eficiente; resultado interpretável.

## Consequências

- Positivas: indicador computável para todos os hexágonos do Brasil em tempo razoável; resultado tem leitura direta ("acesso ponderado por capacidade e distância aos 3 estabelecimentos mais próximos"); winsorização p3-p97 evita que clusters hospitalares de grandes capitais sequestrem a normalização.
- Negativas / trade-offs: não corrige o artefato de cidades pequenas vs grandes; `capacity_score` trata as seis categorias com peso igual, o que é simplificação (cirurgia e ambulatorial têm pesos práticos diferentes na vulnerabilidade real); winsorização p3-p97 é exceção ao padrão p1-p99 (ADR-0012), justificada pelo perfil da distribuição.
- Confiança: Média — modelo defensável e bem-comportado em testes, mas com simplificações conhecidas. Reavaliar 2SFCA se uma versão futura do IIC precisar de comparação intermunicipal (ADR-0007 — atualmente fora de escopo).

## Referências

- ADR-0011 (ponderação dasimétrica), ADR-0012 (normalização padrão), ADR-0016 (exceções à winsorização), ADR-0018 (demais indicadores IV).
- [config/indicators.json](../config/indicators.json) — definição de v4.
- [etl/vulnerability/v4_cnes.py](../etl/vulnerability/v4_cnes.py) — implementação 3-NN gravitacional.
- [etl/discarded/v4_cnes_2sfca.py](../etl/discarded/v4_cnes_2sfca.py) — implementação 2SFCA descartada (preservada para referência).
- [report/methodological_notes.md](../report/methodological_notes.md) — registro técnico de v4.

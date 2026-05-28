# ADR-0043: Separar raw_dir de data_dir e tratar dados brutos como imutáveis

## Status

Accepted — 2026-05-28

## Contexto

O projeto armazena dois tipos de dados com ciclos de vida completamente distintos: dados brutos (inputs originais, externos ao projeto) e dados processados (outputs do próprio pipeline). Até este ponto, ambos eram configurados sob um único `data_dir`, com os dados brutos em `data_dir/inputs/raw/`. Isso forçava ou a duplicar arquivos brutos grandes em uma pasta específica do projeto, ou a misturar dados de origens externas com dados produzidos internamente.

## Decisão

Separar a configuração de caminhos em dois campos independentes em `config/config.local.json`: `data_dir` (para os outputs e inputs processados do projeto) e `raw_dir` (para os dados brutos originais). O `raw_dir` aponta para uma pasta centralizada na máquina da pesquisadora — compartilhada entre projetos — enquanto o `data_dir` permanece exclusivo deste projeto.

Adota-se como princípio que **dados brutos não são alterados**. A pasta `raw_dir` é tratada como fonte somente-leitura: o pipeline apenas lê dados de lá, nunca escreve nem modifica. Qualquer transformação produz um novo arquivo em `data_dir/inputs/clean/` ou em `data_dir/outputs/`. Isso garante rastreabilidade e reprodutibilidade: dado um `raw_dir` e um `data_dir` limpo, o pipeline deve reproduzir todos os resultados do zero.

## Alternativas consideradas

- **Manter tudo em data_dir (descartado)**: obrigava a duplicar dados brutos (muitos GBs) para cada projeto, ou a usar symlinks frágeis. Não resolve o problema de múltiplos projetos consumindo as mesmas fontes.
- **Separar raw_dir de data_dir (escolhido)**: `raw_dir` fica em uma localização estável na máquina, não versionada nem duplicada. `data_dir` contém apenas o que o projeto produziu. Cada campo é configurável independentemente em `config.local.json`.
- **Usar variáveis de ambiente**: possível, mas menos legível e inconsistente com o padrão já adotado pelo projeto de `config.local.json`.

## Consequências

- Positivas: elimina duplicação de dados brutos; deixa claro o que é entrada imutável versus o que o projeto produziu; facilita compartilhar `raw_dir` entre projetos na mesma máquina; torna o pipeline mais explícito sobre suas dependências externas.
- Negativas / trade-offs: `config.local.json` passa a ter dois campos obrigatórios; scripts que não usavam `src.config` precisaram ser corrigidos para importar `cfg.RAW_DIR` ao invés de construir caminhos manualmente.
- Confiança: Alta — a separação reflete uma distinção já existente na prática; a mudança é puramente de configuração e não afeta nenhuma lógica de cálculo.

## Referências

- `src/config.py` — definição de `RAW_DIR` e fallback para `data_dir/inputs/raw` quando `raw_dir` não está presente no config
- `config/config.local.json` e `config/config.example.json` — campos `data_dir` e `raw_dir`
- Scripts corrigidos: `etl/exposure/sgb/00` a `06`, `etl/discarded/07_sgb_analyse_fn_hand.py`, `explore/checks/check_values_IG.py`

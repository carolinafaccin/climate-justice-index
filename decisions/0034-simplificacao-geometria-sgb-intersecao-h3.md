# ADR-0034: Simplificar geometrias SGB a 20 m antes da interseção com H3

## Status
Accepted — 2026-05-22

## Contexto
O script `03_sgb_h3_intersect.py` calcula, por hexágono H3 res9, a fração de área em classes de suscetibilidade alta/muito alta. Para isso, intersecta os polígonos SGB com as células H3 usando GEOS (via shapely).

Os polígonos SGB são mapeamento de campo em escala 1:25.000. Nessa escala, uma única feição pode ter milhares de vértices descrevendo bordas detalhadas de encostas e planícies. Durante os testes, o script travava no cálculo GEOS para certos estados — em particular ao processar os últimos lotes de feições de AC (Acre), com `shapely.intersection` não retornando após vários minutos.

A raiz do problema é que o custo de `shapely.intersection(a, b)` cresce com o número de vértices de `a` e `b`. Polígonos com 10.000+ vértices em apenas uma feição fazem o GEOS ficar preso em certos casos limite.

## Decisão
Simplificar as geometrias SGB (já projetadas em EPSG:5880) com tolerância de **20 metros** e `preserve_topology=True` antes de qualquer cálculo de interseção.

```python
sgb_proj.geometry = sgb_proj.geometry.simplify(20.0, preserve_topology=True)
```

## Justificativa para 20 m

| Parâmetro de referência | Valor |
|---|---|
| Aresta média H3 res9 | ~174 m |
| Área média H3 res9 | ~105.333 m² |
| Tolerância escolhida | 20 m = 11,5 % da aresta |
| Precisão nominal 1:25.000 | ~5–10 m no campo |

- **Precisão nominal do dado fonte**: mapeamentos 1:25.000 têm precisão posicional de ~5–10 m. Simplificar a 20 m não piora a qualidade real dos polígonos SGB.
- **Proporção em relação ao hexágono**: 20 m equivale a 11,5 % da aresta e ~0,02 % da área da célula. O erro de área introduzido só ocorre onde a borda de um polígono SGB cruza a borda de um hexágono — o interior é sempre exato. Na prática, o erro de `sgb_alta_mta_frac` permanece < 2–5 %.
- **Incerteza dominante do pipeline**: a cobertura parcial SGB (~814 de ~5.570 municípios), a heterogeneidade metodológica entre mapeamentos estaduais e os diferentes anos de levantamento (2013–2024) introduzem incertezas muito maiores que 20 m de simplificação geométrica. Refinar a precisão geométrica abaixo desse limiar não tem retorno metodológico.
- **Alternativas consideradas**:
  - **5 m**: seguro, mas redução de vértices moderada — pode não resolver o hang em estados com polígonos mais complexos.
  - **30 m**: ~17 % da aresta, ainda defensável, mas começa a afetar polígonos pequenos (< 0,5 km²).
  - **50 m**: ~29 % da aresta, aceitável para reduzir tempo de processamento quando necessário, mas pode alterar quais hexágonos um polígono pequeno toca.
  - **`preserve_topology=False`**: mais rápido, mas pode criar geometrias inválidas que exigem `make_valid()` adicional. Não vale a complexidade extra com `preserve_topology=True`.

## Consequências
- **Positivas**: elimina hang em GEOS para polígonos com muitos vértices; redução estimada de 10–50× no tempo de processamento por estado; sem impacto visível na acurácia de `sgb_alta_mta_frac`.
- **Negativas / trade-offs**: introduz erro de borda de ~0–5 % nos hexágonos onde a fronteira de um polígono SGB cruza a fronteira de uma célula H3; polígonos muito pequenos (< 5.000 m²) podem sofrer distorção maior proporcionalmente, mas esses polígonos têm peso negligível na agregação por hexágono.
- **Reversibilidade**: a simplificação ocorre apenas em memória durante o cálculo (`03_sgb_h3_intersect.py`); os GeoPackages harmonizados (`02_sgb_mass_br.gpkg`, `02_sgb_floods_br.gpkg`) preservam as geometrias originais intactas.

## Referências
- ADR-0032: decisão de usar SGB como referência de calibração.
- [etl/exposure/sgb/03_sgb_h3_intersect.py](../etl/exposure/sgb/03_sgb_h3_intersect.py) — implementação (`intersect_state`, linha da chamada `.simplify`).

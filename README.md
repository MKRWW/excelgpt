# excelgpt

Ein winziger GPT (Char-Level, Tiny Shakespeare) wird in PyTorch
trainiert, und ein Dump-Skript schreibt fuer einen fest verdrahteten
Test-Prompt JEDES Zwischenergebnis des Forward-Pass als CSV. Zweck dieses
Projekts: ein Sprachmodell-Forward-Pass, der spaeter **in Excel, vollstaendig
in VBA** nachgerechnet wird, mit den Gewichten in Tabellenzellen — ohne
Bibliotheken und ohne Laufzeit-Interpreter. Die CSVs unter `reference/`
sind die Referenz, gegen die eine unabhaengige Reimplementierung Layer fuer
Layer verglichen wird — deshalb ist nicht die Modellqualitaet der Kern,
sondern die Nachvollziehbarkeit und Exaktheit der Konventionen.

Wer das fertige Modell bedienen will: **[PROMPTING.md](PROMPTING.md)** erklaert, warum
Anweisungen nicht funktionieren, welche Prompts stattdessen etwas taugen und was die
Temperatur macht.

## Ausprobieren, ohne etwas zu bauen

Die fertige Arbeitsmappe liegt als **[build/excelgpt.xlsm](build/excelgpt.xlsm)** im
Repository. Herunterladen genuegt, es wird weder Python noch ein Training gebraucht.

Drei Schritte, der erste ist der, an dem es sonst scheitert:

1. **Datei entsperren.** Rechtsklick auf die heruntergeladene Datei, *Eigenschaften*,
   unten *Zulassen* ankreuzen, *OK*. Excel blockiert Makros in Dateien aus dem Internet
   seit 2022 vollstaendig — ohne diesen Schritt oeffnet die Mappe zwar, aber der Knopf
   tut nichts, und es erscheint nur ein gelber Balken.
2. **Oeffnen und Makros zulassen.** Die Mappe startet auf dem Bedienpult `00_LLM`.
3. **Prompt eintragen und auf Generate druecken.** Gute Prompts stehen in
   [PROMPTING.md](PROMPTING.md); `ROMEO:` ist ein sicherer Anfang.

Vorausgesetzt wird Windows mit Excel Desktop. Die Rechnung steckt in VBA, das es in
Excel im Browser nicht gibt.

Wer lieber selbst baut, findet den Weg unter [Setup](#setup) und
[Arbeitsmappe](#arbeitsmappe) — dann trainiert das Modell auf dem eigenen Rechner neu.

## Setup

```
uv sync
```

Danach die vier Kommandos in dieser Reihenfolge:

```
uv run python data/prepare.py
uv run python train.py
uv run python sample.py --seed 1337
uv run python dump_reference.py
```

- `data/prepare.py` laedt Tiny Shakespeare, baut das Vokabular (`data/meta.json`,
  `vocab_size == 65`) und splittet 90/10 in `data/train.bin` / `data/val.bin`
  (uint16).
- `train.py` trainiert (AdamW, warmup + Cosine) und schreibt `out/ckpt.pt`.
- `sample.py` erzeugt Text; bei identischem `--seed` ist die Ausgabe
  byte-identisch (Seed ueber einen expliziten `torch.Generator`).
- `dump_reference.py` schreibt fuer den Prompt `To be, or not to be`
  (T = 19) alle Zwischenergebnisse als CSV nach `reference/` plus
  `manifest.csv`, mit vier Selbstpruefungen (Abbruch mit Exit-Code 1 bei
  Verletzung).

## Architektur (verbindlich)

Pre-LayerNorm-Transformer:

```
x = tok_emb[idx] + pos_emb[0..T-1]
fuer jeden Block:
    x = x + attn(ln1(x))
    x = x + mlp(ln2(x))
logits = lm_head(ln_f(x))
```

Hyperparameter (Defaults im Code, per CLI ueberschreibbar):

| Name       | Wert |
|------------|------|
| n_layer    | 4    |
| n_head     | 4    |
| n_embd     | 128  |
| head_dim   | 32 (= n_embd / n_head) |
| block_size | 64   |
| vocab_size | aus data/meta.json (Tiny Shakespeare: 65) |
| mlp_hidden | 512 (= 4 * n_embd) |
| dropout    | 0.0  |

## Konventionen (Punkte 1–12, verbindlich)

1. **GELU = tanh-Approximation**, exakt
   `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
   In PyTorch: `nn.GELU(approximate="tanh")`. KEINE erf-Variante.
2. **LayerNorm**: eps = 1e-5, Varianz ohne Bessel-Korrektur (biased), mit
   weight UND bias.
3. **Jedes `nn.Linear` hat `bias=True`.** Keine Ausnahme. Die Regel "jede
   lineare Schicht ist weight + bias" gilt einheitlich.
4. **Kein Weight-Tying.** `lm_head.weight` ist ein eigener Parameter,
   unabhaengig von `wte.weight`.
5. **Gelernte absolute Positions-Embeddings** (`wpe`), Position = Index im
   Kontextfenster (0..T-1), nicht Prompt-relativ. Kein RoPE, kein ALiBi.
6. **Attention explizit ausgeschrieben.** `F.scaled_dot_product_attention` und
   jede Form von Flash-/Memory-Efficient-Attention sind verboten, weil die
   Zwischenergebnisse sichtbar bleiben muessen.
7. **QKV-Projektion**: ein einziges `nn.Linear(n_embd, 3 * n_embd)`. Die
   Ausgabe wird in der Reihenfolge **[Q | K | V]** gesplittet (Q = Spalten
   0..127, K = 128..255, V = 256..383). Innerhalb jedes Blocks belegt Head h die
   Spalten `h*head_dim .. h*head_dim + head_dim - 1`.
8. **Attention-Skalierung**: `scores = (Q @ K^T) / sqrt(head_dim)`, also
   / sqrt(32).
9. **Causal Mask**: `scores[t, s]` wird maskiert fuer `s > t`. Im CSV-Dump wird
   der maskierte Wert als `-1.0e30` geschrieben, NICHT als `-inf` (CSV muss
   zahlenrein sein). Intern darf negativ-unendlich verwendet werden, aber der
   Dump ersetzt es.
10. **Softmax numerisch stabil**: Maximum der Zeile abziehen, dann exp, dann
    durch die Summe teilen.
11. **Dropout ist ueberall 0.0** und wird gar nicht erst als Modul eingebaut.
12. Alles rechnet in **float32**. Kein autocast, kein bf16, kein TF32-Matmul
    (`torch.backends.cuda.matmul.allow_tf32 = False` und
    `torch.backends.cudnn.allow_tf32 = False` setzen, sonst weichen
    GPU-Ergebnisse im vierten Nachkommastellenbereich ab).

## Dump-Konventionen (verbindlich)

- Ein CSV pro Tensor, Zielverzeichnis `reference/`.
- **Keine Kopfzeile, keine Indexspalte.** Reine Zahlenmatrix.
- Feldtrenner: Komma. Dezimaltrenner: Punkt.
- Zahlenformat: `%.10e` fuer alle Fliesskommawerte. Ganzzahlen (Token-IDs) als
  reine Ziffernfolge ohne Exponent.
- **Zeilen = Zeitschritte t (0..T-1, aufsteigend), Spalten = Feature-Index
  (aufsteigend).** Bei Attention-Matrizen: Zeile = Query-Position t, Spalte =
  Key-Position s.
- Kein `nan`, kein `inf`, kein negativ-unendlich in irgendeiner Datei.
- `reference/manifest.csv` ist die einzige Datei mit einer Kopfzeile:
  `name,rows,cols,description` — eine Zeile pro geschriebenem CSV (Dateiname
  ohne `.csv`), aufsteigend sortiert nach Schreibreihenfolge.

## Trace-Keys (exakte Dateinamen)

`l` laeuft 0..n_layer-1, `h` laeuft 0..n_head-1. T = Anzahl Prompt-Zeichen,
C = 128.

Global:

| Datei | Form | Inhalt |
|---|---|---|
| `00_tokens`   | T x 1  | Token-IDs des Prompts (Ganzzahlen) |
| `01_tok_emb`  | T x C  | wte[idx] |
| `02_pos_emb`  | T x C  | wpe[0..T-1] |
| `03_x_input`  | T x C  | Summe aus 01 und 02 |

Pro Layer l (hier n_layer = 4, n_head = 4; 37 Dateien pro Layer):

| Datei | Form | Inhalt |
|---|---|---|
| `L{l}_10_ln1`                 | T x C   | LayerNorm 1 |
| `L{l}_11_q_h{h}`              | T x 32  | Q von Head h |
| `L{l}_12_k_h{h}`              | T x 32  | K von Head h |
| `L{l}_13_v_h{h}`              | T x 32  | V von Head h |
| `L{l}_14_scores_scaled_h{h}`  | T x T   | Q@K^T / sqrt(32), UNmaskiert |
| `L{l}_15_scores_masked_h{h}`  | T x T   | wie 14, maskierte Felder = -1.0e30 |
| `L{l}_16_attn_probs_h{h}`     | T x T   | nach Softmax (Zeilensumme 1) |
| `L{l}_17_head_out_h{h}`       | T x 32  | probs @ V |
| `L{l}_18_attn_concat`         | T x C   | Heads in Reihenfolge h=0..3 nebeneinander |
| `L{l}_19_attn_proj`           | T x C   | nach Output-Projektion inkl. bias |
| `L{l}_20_resid_post_attn`     | T x C   | Residual nach Attention |
| `L{l}_30_ln2`                 | T x C   | LayerNorm 2 |
| `L{l}_31_fc`                  | T x 512 | erste MLP-Schicht inkl. bias |
| `L{l}_32_gelu`                | T x 512 | nach GELU |
| `L{l}_33_mlp_proj`            | T x C   | zweite MLP-Schicht inkl. bias |
| `L{l}_34_resid_post_mlp`      | T x C   | Residual nach MLP |

Abschluss:

| Datei | Form | Inhalt |
|---|---|---|
| `90_ln_f`             | T x C  | finaler LayerNorm |
| `91_logits`           | T x V  | Logits fuer alle Positionen |
| `92_logits_last`      | 1 x V  | Logits der letzten Position |
| `93_probs_last_temp1` | 1 x V  | Softmax davon, Temperatur 1.0 |

Zahl der Dateien: 4 global + 4 Layer x 37 (1 ln1 + 4 Heads x 7 + 8 weitere) +
4 Abschluss = **156 CSV** plus `manifest.csv`. (Die Rechnung "33 pro Layer ->
140" im Aufgabenblatt zaehlt die pro-Head-Gruppen zusammengefasst; die
konkrete Dateiliste oben ist verbindlich, weil sie die exakten Trace-Key-
Dateinamen festlegt.)

## Export-Konventionen (gueltig ab dem Gewichts-Export)

Gewichte von `nn.Linear` liegen in PyTorch als `(out_features, in_features)`
und werden beim Export **transponiert** abgelegt, also als
`(in_features, out_features)`, damit die Zielumgebung `y = x @ W + b` ohne
Transposition rechnen kann. Bias bleibt Zeilenvektor `(1, out_features)`.

## Locale-Hinweis

Die CSVs verwenden den Punkt als Dezimaltrenner. Beim Import in eine
Tabellenkalkulation mit deutscher Locale (die den Komma als Dezimaltrenner
erwartet) muss dies entsprechend behandelt werden: sonst werden die
`%.10e`-Werte falsch geparst (z. B. wird `1.2345678900e+00` als Zahl mit
Tausenderpunkt statt als Fließkommazahl interpretiert).

## Arbeitsmappe

Nach dem Dump kommen die Gewichte in eine Excel-Arbeitsmappe. Die vier
Kommandos in dieser Reihenfolge:

```
uv run python export_weights.py
uv run python build_workbook.py
uv run python inject_vba.py
uv run python verify_vba.py
```

- `export_weights.py` schreibt die 54 Gewichtstensoren des Checkpoints als
  CSV-Dateien nach `export/` (dazu `manifest.csv`, `vocab.csv` und
  `config.csv`).
- `build_workbook.py` legt diese CSVs als die benannten Bereiche in die
  Arbeitsmappe `build/excelgpt.xlsm` ab.
- `inject_vba.py` importiert die fuenf Makromodule aus `vba/` in das
  VBA-Projekt der Arbeitsmappe (Austausch statt Hinzufuegen, dazu die
  Blaetter `97_Trace` und `98_Probe`) und speichert das Ergebnis als `.xlsm`.
  Das Skript setzt im Trust Center die Einstellung "Zugriff auf das
  VBA-Projektobjektmodell vertrauen" voraus; ohne sie bricht es mit einem
  Rechtefehler ab.
- `verify_vba.py` fuehrt den Vorwaertsdurchlauf mit Mitschnitt aller
  Zwischenergebnisse in der Arbeitsmappe aus und vergleicht jeden Tensor
  gegen `reference/`.

Nach `inject_vba.py` ist die Datei fertig: man oeffnet `build/excelgpt.xlsm`,
landet auf dem Bedienpult, traegt einen Prompt ein und drueckt Generate.

Blattaufteilung (Blattreihenfolge exakt wie in der Tabelle):

| Blatt | Inhalt |
|---|---|
| `00_LLM` | Bedienpult — in dieser Stufe nur anlegen, leer lassen. |
| `10_Embedding` | Token- und Positions-Embeddings (`wte`, `wpe`). |
| `20_Layer0` | Alle 12 Tensoren von Layer 0. |
| `30_Layer1` | Alle 12 Tensoren von Layer 1. |
| `40_Layer2` | Alle 12 Tensoren von Layer 2. |
| `50_Layer3` | Alle 12 Tensoren von Layer 3. |
| `90_Output` | Finaler LayerNorm und `lm_head` (transponiert) mit Bias. |
| `99_Meta` | Konfigurationswerte und Vokabular. |

Alle benannten Bereiche sind arbeitsmappenweit (workbook scope), Grossbuchstaben,
keine Blattpraefixe. Die 62 Bereiche in Vertragsgestaltung:

| Name | Form | Quelle im state_dict |
|---|---|---|
| `WTE` | 65 x 128 | `transformer.wte.weight` |
| `WPE` | 64 x 128 | `transformer.wpe.weight` |
| `L0_LN1_W` | 1 x 128 | `transformer.h.0.ln1.weight` |
| `L0_LN1_B` | 1 x 128 | `transformer.h.0.ln1.bias` |
| `L0_ATTN_W` | 128 x 384 | `transformer.h.0.attn.c_attn.weight`, transponiert |
| `L0_ATTN_B` | 1 x 384 | `transformer.h.0.attn.c_attn.bias` |
| `L0_PROJ_W` | 128 x 128 | `transformer.h.0.attn.c_proj.weight`, transponiert |
| `L0_PROJ_B` | 1 x 128 | `transformer.h.0.attn.c_proj.bias` |
| `L0_LN2_W` | 1 x 128 | `transformer.h.0.ln2.weight` |
| `L0_LN2_B` | 1 x 128 | `transformer.h.0.ln2.bias` |
| `L0_FC_W` | 128 x 512 | `transformer.h.0.mlp.c_fc.weight`, transponiert |
| `L0_FC_B` | 1 x 512 | `transformer.h.0.mlp.c_fc.bias` |
| `L0_FCPROJ_W` | 512 x 128 | `transformer.h.0.mlp.c_proj.weight`, transponiert |
| `L0_FCPROJ_B` | 1 x 128 | `transformer.h.0.mlp.c_proj.bias` |
| `L1_LN1_W` | 1 x 128 | `transformer.h.1.ln1.weight` |
| `L1_LN1_B` | 1 x 128 | `transformer.h.1.ln1.bias` |
| `L1_ATTN_W` | 128 x 384 | `transformer.h.1.attn.c_attn.weight`, transponiert |
| `L1_ATTN_B` | 1 x 384 | `transformer.h.1.attn.c_attn.bias` |
| `L1_PROJ_W` | 128 x 128 | `transformer.h.1.attn.c_proj.weight`, transponiert |
| `L1_PROJ_B` | 1 x 128 | `transformer.h.1.attn.c_proj.bias` |
| `L1_LN2_W` | 1 x 128 | `transformer.h.1.ln2.weight` |
| `L1_LN2_B` | 1 x 128 | `transformer.h.1.ln2.bias` |
| `L1_FC_W` | 128 x 512 | `transformer.h.1.mlp.c_fc.weight`, transponiert |
| `L1_FC_B` | 1 x 512 | `transformer.h.1.mlp.c_fc.bias` |
| `L1_FCPROJ_W` | 512 x 128 | `transformer.h.1.mlp.c_proj.weight`, transponiert |
| `L1_FCPROJ_B` | 1 x 128 | `transformer.h.1.mlp.c_proj.bias` |
| `L2_LN1_W` | 1 x 128 | `transformer.h.2.ln1.weight` |
| `L2_LN1_B` | 1 x 128 | `transformer.h.2.ln1.bias` |
| `L2_ATTN_W` | 128 x 384 | `transformer.h.2.attn.c_attn.weight`, transponiert |
| `L2_ATTN_B` | 1 x 384 | `transformer.h.2.attn.c_attn.bias` |
| `L2_PROJ_W` | 128 x 128 | `transformer.h.2.attn.c_proj.weight`, transponiert |
| `L2_PROJ_B` | 1 x 128 | `transformer.h.2.attn.c_proj.bias` |
| `L2_LN2_W` | 1 x 128 | `transformer.h.2.ln2.weight` |
| `L2_LN2_B` | 1 x 128 | `transformer.h.2.ln2.bias` |
| `L2_FC_W` | 128 x 512 | `transformer.h.2.mlp.c_fc.weight`, transponiert |
| `L2_FC_B` | 1 x 512 | `transformer.h.2.mlp.c_fc.bias` |
| `L2_FCPROJ_W` | 512 x 128 | `transformer.h.2.mlp.c_proj.weight`, transponiert |
| `L2_FCPROJ_B` | 1 x 128 | `transformer.h.2.mlp.c_proj.bias` |
| `L3_LN1_W` | 1 x 128 | `transformer.h.3.ln1.weight` |
| `L3_LN1_B` | 1 x 128 | `transformer.h.3.ln1.bias` |
| `L3_ATTN_W` | 128 x 384 | `transformer.h.3.attn.c_attn.weight`, transponiert |
| `L3_ATTN_B` | 1 x 384 | `transformer.h.3.attn.c_attn.bias` |
| `L3_PROJ_W` | 128 x 128 | `transformer.h.3.attn.c_proj.weight`, transponiert |
| `L3_PROJ_B` | 1 x 128 | `transformer.h.3.attn.c_proj.bias` |
| `L3_LN2_W` | 1 x 128 | `transformer.h.3.ln2.weight` |
| `L3_LN2_B` | 1 x 128 | `transformer.h.3.ln2.bias` |
| `L3_FC_W` | 128 x 512 | `transformer.h.3.mlp.c_fc.weight`, transponiert |
| `L3_FC_B` | 1 x 512 | `transformer.h.3.mlp.c_fc.bias` |
| `L3_FCPROJ_W` | 512 x 128 | `transformer.h.3.mlp.c_proj.weight`, transponiert |
| `L3_FCPROJ_B` | 1 x 128 | `transformer.h.3.mlp.c_proj.bias` |
| `LNF_W` | 1 x 128 | `transformer.ln_f.weight` |
| `LNF_B` | 1 x 128 | `transformer.ln_f.bias` |
| `LM_W` | 128 x 65 | `lm_head.weight`, transponiert |
| `LM_B` | 1 x 65 | `lm_head.bias` |
| `CFG_N_LAYER` | 1 x 1 | Konfigurationswert (4) |
| `CFG_N_HEAD` | 1 x 1 | Konfigurationswert (4) |
| `CFG_N_EMBD` | 1 x 1 | Konfigurationswert (128) |
| `CFG_HEAD_DIM` | 1 x 1 | Konfigurationswert (32) |
| `CFG_BLOCK_SIZE` | 1 x 1 | Konfigurationswert (64) |
| `CFG_VOCAB_SIZE` | 1 x 1 | Konfigurationswert (65) |
| `CFG_MLP_HIDDEN` | 1 x 1 | Konfigurationswert (512) |
| `VOCAB` | 65 x 1 | Zeile i+1 enthaelt den **Codepoint** des Zeichens zu Token-ID i |

`VOCAB` haelt Zahlen, nicht Zeichen. Ein Zellwert, der mit einem Apostroph
beginnt, wird von Excel als Textmarke verschluckt — Token 5 (`'`) kaeme leer
zurueck, und zwar lautlos. Codepoints machen ausserdem die beiden unsichtbaren
Eintraege lesbar: Token 0 ist der Zeilenumbruch (10), Token 1 das Leerzeichen
(32). VBA dekodiert mit `Chr$()`.

VBA adressiert die Gewichte ausschliesslich ueber diese benannten Bereiche,
nie ueber Zelladressen — ein Umbenennen eines Bereichs bricht den Port. Die
Arbeitsmappe wird als `.xlsm` gespeichert, damit in der naechsten Stufe
VBA-Code hinzukommen kann.

## Makrocode

Fuenf Module, in der Reihenfolge, wie `inject_vba.py` sie in das Projekt
importiert:

| Modul | Zweck |
|---|---|
| `Mat` | Matrix-Grundbausteine und blockweiser Zellzugriff. |
| `Nn` | LayerNorm, Softmax, GELU, kausale Maske. |
| `Gpt` | Gewichts-Cache, Embedding-Lookup, Attention, Bloecke, Durchlauf. |
| `Sampler` | Ziehen mit Temperatur, autoregressive Schleife. |
| `Probe` | Einzeleinstiege fuer die Pruefung. |

Der Code liegt als Text unter `vba/` und wird ausschliesslich vom Skript
`inject_vba.py` in die Arbeitsmappe gebracht. Nichts wird im Editor getippt
— sonst waere der Stand der Arbeitsmappe nicht mehr aus dem Repository
herstellbar.

Array-Konvention, die im gesamten Makrocode gilt: 1-basierte Double-Arrays
der Form (Zeile, Spalte), Zeile = Zeitschritt. Gegenueber dem
Referenzmodell, das ab null zaehlt, gilt die Verschiebung um eins:
Token-ID `i` steht in Zeile `i+1`.

Regel zum Zellzugriff: immer ein ganzer Bereich auf einmal in ein Array,
gerechnet wird im Speicher, geschrieben wird einmal. Die Gewichte werden
**einmal** geladen und gehalten; sie je Token neu zu lesen waeren 818241
Zellzugriffe pro Schritt.

## Bedienpult

`00_LLM` ist das Bedienpult, alle uebrigen Blaetter sind Datenblaetter.

Das gesamte Layout entsteht in `Ui.SetupSheet`, nicht von Hand. Es ist damit
versionierbar und nach jedem Neubau identisch. Wer etwas verschieben will,
aendert den Code, nicht die Zellen. `inject_vba.py` ruft das Makro nach dem
Import auf.

Tabelle der Bedienelemente. Auch hier gilt: angesprochen wird ueber die
Namen, nie ueber Zelladressen:

| Name | Zweck |
|---|---|
| `UI_PROMPT` | Der Prompt, den das Modell fortsetzt. |
| `UI_TEMP` | Temperatur beim Ziehen: klein = braver, gross = wilder. |
| `UI_TOKENS` | Anzahl der Zeichen, die erzeugt werden. |
| `UI_HEAD` | Welcher Aufmerksamkeitskopf in der Heatmap gezeigt wird, 0 bis 3. |
| `UI_SEED` | Startwert: gleicher Startwert und gleiche Eingaben ergeben denselben Text. |
| `UI_STATUS` | Der laufende Stand der Erzeugung. |
| `UI_OUTPUT` | Der bisher erzeugte Text. |
| `UI_HEATMAP` | Das Aufmerksamkeitsraster des letzten Layers. |

Der Knopf ist eine Form mit `OnAction = "Ui.Generate"`.

Die Heatmap zeigt die Aufmerksamkeit des **letzten** Layers fuer den in
`UI_HEAD` gewaehlten Kopf, als 64 mal 64 Raster mit einer Farbskala als
bedingte Formatierung. Zeile = das Zeichen, das gerade dran ist; Spalte =
worauf es zurueckschaut. Dass nur das untere Dreieck gefuellt ist, ist die
kausale Maske.

Alle Spalten sind schmal und gleich breit, damit das Raster quadratisch wird;
Beschriftungen und Eingabefelder ueberspannen deshalb mehrere Spalten.

Die Erzeugungsschleife liegt in `Ui.Generate` und nicht in `Sampler.Generate`,
weil sie nach jedem Token Ausgabe, Status und Heatmap auffrischt und
`DoEvents` aufruft. `Sampler.Generate` bleibt die Variante ohne Blattverkehr,
die die Pruefung benutzt.

Gemessenes Tempo: rund eine halbe Sekunde je Token bei kurzem Kontext, gegen
eine Sekunde, wenn das Kontextfenster voll ist.

## Verifikation

`verify_vba.py` prueft in zwei Stufen: erst jeden Baustein einzeln ueber
das Probe-Modul, dann den gesamten Stapel gegen alle 156 Zwischentensoren
aus `reference/`.

Standard-Toleranz 1e-4, ueber `--tol` aenderbar; `--prompt` waehlt den
Prompt (Standard ist der des Dumps: `To be, or not to be`), `--verbose`
listet jeden Tensor statt nur der auffaelligen. Gemessener Stand: groesste
Abweichung 1.1e-06 bei den Bausteinen und 6.1e-06 ueber den ganzen Stapel.

Bei einem Fehlschlag nennt das Skript den **fruehesten** abweichenden
Tensor in Rechenreihenfolge, nicht den mit der groessten Abweichung.
Alles hinter einer falschen Zwischenstufe ist Folge davon; wer die groesste
Abweichung jagt, sucht typischerweise mehrere Schichten hinter der
Ursache.

Exit-Code 1 bei jeder Abweichung, damit sich das Skript als Gate verwenden
laesst.

`inject_vba.py` prueft die Quellen, bevor die Tabellenkalkulation ueberhaupt
startet, und weist Variablen zurueck, die nie deklariert wurden, sowie Namen,
die mit reservierten Woertern kollidieren. Der Grund gehoert dazu: uebersetzt
wird prozedurweise, ein solcher Fehler faellt sonst erst auf, wenn die
betroffene Prozedur zum ersten Mal laeuft.

## Fallstricke

1. **`Scale` ist ein reserviertes Wort.** `Dim scale As Double` ist ein
   Syntaxfehler.
2. **Uebersetzt wird prozedurweise.** Ein Syntaxfehler in einer Funktion
   faellt erst auf, wenn sie zum ersten Mal aufgerufen wird — ein
   Rauchtest muss deshalb einmal durch den gesamten Stapel laufen, nicht
   nur eine Hilfsfunktion aufrufen.
3. **Der Funktionsname in einer Argumentliste** wird als rekursiver Aufruf
   gelesen, nicht als Rueckgabewert. Zwischenwerte brauchen eine eigene
   Variable.
4. **Ein verbundener Bereich liefert ueber `Value2` ein Array**, keinen
   Einzelwert. Jede Umwandlung daraus scheitert mit "Typen unvertraeglich".
   Bedienelemente werden deshalb ueber ihre linke obere Zelle gelesen.
5. **Das Gitternetz haengt am Fenster**, nicht am Blatt. Das Fenster, in dem
   das Layout gebaut wird, ist nicht das, in dem spaeter jemand sitzt —
   deshalb wird die Ansicht beim Oeffnen der Datei gesetzt.
6. **Beim Loeschen aus einer Collection waehrend der Iteration** ueberspringt
   die Sprache Eintraege. Rueckwaerts ueber den Index laufen.

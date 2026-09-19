# TODO

Ordine vincolante: §10 della specifica. `→` = dipende da. Dettaglio e criteri in `plan.md`.

## Fase A — nessuno store, nessun training

- [x] **M0** `legcheck` sulla cella argmax (0,2 / 12) — **chiusa in negativo 2026-09-19** — *nel repo TradingVision* · 5 min
      ↳ `legsweep` ha spazzato 117 celle dei due numeri liberi dell'etichetta; la superficie non ha
        ottimo interno e alla cella migliore `edge` è 0,0330 contro 0,0298 — tre semi di rumore.
        M0 chiede al **prezzo** se quel vincitore è vero
      ✔ decili quotati contro il prezzo con |t|; verdetto scritto in §5
      ✔ **esito: piatta.** ic_raw −0,0194 a 6 barre, max |t| = 1,68 su 30 decili, nessuna
        monotonia. Rank IC 0,4753 contro l'etichetta. Verdetto in §5 dello schema
      ✘ la griglia non aveva un vincitore. **Non si rifà sulla Rank IC**
- [x] **M1a** ✅ E1, 2026-09-19 — **esposizioni scorrelate → `E1`** · 3 h · *il costo non entra qui*
      ↳ il dataset decide cosa il modello può imparare; il costo decide con quale strumento lo si
        compra (M1b) e quanto costa simularlo (M2)
      ✔ serve solo storico **daily** dei candidati, non lo store intraday
      ✔ venti esposizioni + matrice di correlazione + una motivazione per ciascuna
      ✔ **esito: corr media 0,3537 in stima → 0,3517 dopo. sd_t +26% / +22% contro i soli settori**
      ✔ contenimento KRE⊂XLF e XBI⊂XLV risolto in favore del contenitore
      ✔ stimata **sul primo fold**, verificata e non rifatta sugli altri (§8: chi decide, decide
        sul train)
- [x] **M1b** ✅ U1, 2026-09-19 — strumento per esposizione — spread effettivo → **U1** · 3 h → M1a
      ✔ tabella candidati (spread mediano ponderato RTH, esclusa prima/ultima mezz'ora) + costo bp
      ✔ `SYMBOLS` = **venti** strumenti, ognuno col commento che dice **quale esposizione implementa**
      ✘ nessuno strumento accettabile in un'esposizione → si cambia strumento o la si dichiara non
        tradabile; **non** la si sostituisce con una più economica ma correlata a un'altra
- [x] **M2** ✅ `costs.py`, fee per simbolo · 3 h → M1b *(i numeri; il codice si scrive prima)*
      ✔ il suo posto è la **simulazione**, non il dataset
      ✔ **esito: la tabella di §2 si riproduce a 1e-3 bp**; lo scarto è l'arrotondamento della
        tabella (somma di componenti già arrotondate), non la formula
      ✔ `0,103 + 0,975/P + 5000·s/P`; SEC §31 e TAF in config, mai costanti
      ✔ dividendo su short, per simbolo, **solo gamba corta**
      ✔ self-check: a fee piatta i numeri del progetto precedente tornano **identici**
- [x] **Checkpoint A** ✅ **raggiunto** — E1 e U1 definiti e **distinti**, costo per simbolo confinato alle
      simulazioni, M0 con verdetto

## Fase B — store e tempo di sessione

- [x] **M3a** ✅ *anticipata: IEX daily parte dal 2018 con buchi, M1a non era misurabile* — `ALPACA_FEED` default **`sip`** (misurato: gratuito, dal 2016) · 5 min
- [ ] **M3** `data.alpaca_equities` + `calendar` (NYSE) · 4 h → M1b
      ✔ `feed=sip`, `adjustment=all`, base **1Min**, RTH esplicita, dal 2016-01-04
      ✔ stamp della cache che rifiuta parametri diversi
      ✔ le **tre trappole** hanno una risposta misurata nel docstring: DST (UTC vs ET),
        mezze giornate, retroattività di `adjustment=all`
- [ ] **M3.5** dispersione cross-sezionale `sd_t` · 1 h → M3 · **prima della M5**
      ↳ non è un doppione di M1a: quella è progetto su rendimenti daily, questa è validazione alla
        frequenza di trading, dove il fattore comune domina di più. Un E1 buono può fallire qui
      ✔ distribuzione per data su U1 a 1m/5m/15m, confrontata col crypto
      ✘ un ordine sotto il crypto → l'etichetta non ha materia prima
- [ ] **M4** **tempo di sessione** — indice locale, timestamp degradato a colonna · 1 g → M3
      ✔ decisione di regime: **(a) intraday-only, flat 16:00** (raccomandata)
      ✔ tre rami 1m/5m/15m, griglia base 1m (il 15m a N=24 è il tetto)
      ✔ **test per troncamento**: si tronca a una chiusura, nessuna colonna della sessione
        successiva cambia
      ✔ test di anticipazione invariato: alle 10:05 il ramo 15m vede la barra chiusa alle 10:00
      ✔ nessun ffill e nessuna rolling attraversa una chiusura
- [ ] **Checkpoint B** — store riproducibile, M4 testata. Senza M4 ogni misura dopo è sospetta

## Fase C — il go/no-go

- [ ] **M5** `exhaustcheck --price` + baseline `factor` su U1, costo vero, **nessun training** · 1 h
      → M2, M4
      ✔ controllo a **esposizione fissa** (`rotation_null`), mai buy-and-hold
      ✘ lordo ≤ 0 contro `rotation_null` → **nessun modello lo salva**. Fine del progetto
- [ ] **Checkpoint C** — **go/no-go**

## Fase D — etichetta, leakage, riferimento

- [ ] **M6** `oracle`: lag 0 e lag W, `finestra_estremi` rimisurata · 2 h → M5
      ✔ vincolo nuovo: a `L = finestra` la gamba sta **dentro la sessione**
      ✔ orizzonte di purging rimisurato in **barre di sessione** (il **max** è il numero che conta)
- [ ] **M7** `linear` — allarme leakage · 1 h → M6
      ✔ atteso **Rank IC ≈ 0** · ✘ Rank IC alta → leakage di sessione, **si torna alla M4**
- [ ] **M8** `gbm --horizon` + `selection` rifatta su U1 · 4 h → M7
      ✔ set di feature come **parametro** (`--features`), non cancellazione
      ✔ stagionalità intraday: confrontare (i) normalizzazione per fascia oraria e (ii) seno/coseno
        di sessione — **misurare, non scegliere a priori**
      ✔ **baseline a sola ora del giorno** come controllo obbligatorio

## Fase E — il modello

- [ ] **M9** `swing` due stadi, 4 fold, 5 seed, δ rimisurato · 1 g → M8
      ✔ sbarramenti in ordine: `legcheck` (|t| > 2, monotono) → `rotation_null` (z > 2, ≥ 500 rot.)
        → netto col costo **per simbolo** → **information ratio** → dispersione sui 4 fold
      ✘ mai promuovere sulla Rank IC contro l'etichetta (0,4114 vs **−0,0405** sul prezzo)
- [ ] **M10** paper: **locate sugli ETB** e slippage contro il mid · 1 sett. → M9
      ✘ locate per ogni ingresso short → always-in non eseguibile, il libro diventa long-only +
        hedge di indice: **strategia diversa, da riprezzare**

## Per ogni slice

- [ ] self-check in fondo al modulo + iscritto in `tests/test_selfchecks.py`
- [ ] docstring: cosa misura, che numero è uscito, quale alternativa è stata scartata e perché
- [ ] `ruff` · `black` · `pytest` verdi
- [ ] sezione corrispondente di `equity_dataset_schema.html` aggiornata
- [ ] commit il cui soggetto è **la misura**, non il file

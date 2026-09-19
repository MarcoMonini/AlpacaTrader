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
      ✔ `universe.U1` = venti strumenti, ognuno col commento che dice quale esposizione implementa
      ✔ **esito: l'ipotesi Vanguard di §4 è morta** — VPU 7¢, VDC 9¢, VAW 12¢, VHT 16¢, tutti
        sopra il loro pareggio. La colonna «pareggio» di §4 ha previsto tutte e 7 le decisioni
      ✔ costo U1 da 0,281 bp (IWM) a 2,170 (XHB) — **7,7×**, il vettore per simbolo resta necessario
      ⚠ XHB 2,170 · XOP 1,927 · UUP 1,899 stanno **sopra 1,76 bp**, il miglior pareggio mai misurato
      ✘ nessuno strumento accettabile in un'esposizione → si cambia strumento o la si dichiara non
        tradabile; **non** la si sostituisce con una più economica ma correlata a un'altra
- [x] **M2** ✅ `costs.py`, fee per simbolo · 3 h → M1b *(i numeri; il codice si scrive prima)*
      ✔ il suo posto è la **simulazione**, non il dataset
      ✔ **esito: la tabella di §2 si riproduce a 1e-3 bp**; lo scarto è l'arrotondamento della
        tabella (somma di componenti già arrotondate), non la formula
      ✔ **esito: la tabella di §2 si riproduce a 1e-3 bp**; lo scarto è l'arrotondamento della
        tabella (somma di componenti già arrotondate), non la formula
      ✔ `0,103 + 0,975/P + 5000·s/P`; SEC §31 e TAF in config, mai costanti
      ✔ dividendo su short, per simbolo, **solo gamba corta**
      ✔ self-check: a fee piatta i numeri del progetto precedente tornano **identici**
- [x] **Checkpoint A** ✅ **raggiunto** — E1 e U1 definiti e **distinti**, costo per simbolo confinato alle
      simulazioni, M0 con verdetto

## Fase B — store e tempo di sessione

- [x] **M3a** ✅ *anticipata: IEX daily parte dal 2018 con buchi, M1a non era misurabile* — `ALPACA_FEED` default **`sip`** (misurato: gratuito, dal 2016) · 5 min
- [x] **M3** ✅ **2026-09-19** — `data.store` + `calendar` (NYSE) · → M1b
      ✔ `feed=sip`, `adjustment=all`, base **1Min**, RTH esplicita, dal 2016-01-04
      ✔ stamp della cache che rifiuta parametri diversi
      ✔ **2.693 sessioni, 2016-01-04 → 2026-09-18, ~20,9M barre, 531 MB, 20/20 simboli completi**
      ✔ trappola 1 (DST): apertura 09:30 locali sempre, UTC spostata 21 volte — nessuna sorpresa
      ✔ trappola 2 (mezze giornate): **non** si fermano alle 13:00, il nastro stampa fino alle 15:59.
        Rilevate dal pomeriggio vuoto e **a maggioranza fra simboli**: 21, cioè 2,0/anno
      ✔ trappola 3 (retroattività): **ha cambiato un parametro** — `adjustment=split`, non `all`
      ⚠ **trappola nuova**: l'intervallo restituito dipende dalla fine richiesta. Chunk mensili +
        `verify` che conta le sessioni. JNK aveva perso 24 sessioni senza alcun errore
      ✔ 7 sessioni corte su 2.693, di cui 4 sono i circuit breaker del marzo 2020
- [x] **M3.5** ✅ **passata 2026-09-19** — dispersione `sd_t` · presa **prima della M3**
      ↳ non è un doppione di M1a: quella è progetto su rendimenti daily, questa è validazione alla
        frequenza di trading, dove il fattore comune domina di più. Un E1 buono può fallire qui
      ✔ distribuzione per data su U1 a 1m/5m/15m, confrontata col crypto
      ✔ **esito: 1,4×–2,1× sotto il crypto, non un ordine di grandezza. L'etichetta ha materia**
      ✔ non serviva lo store: finestre campionate su tre regimi (2018 / 2022 / 2026)
      ⚠ il regime **recente è il più povero** (0,000585 a 5m contro 0,000891 nel 2022)
      ✔ la dispersione cresce come √barra: i membri si muovono quasi indipendentemente
- [x] **M4** ✅ **2026-09-19** — tempo di sessione, indice locale, timestamp a colonna → M3
      ✔ decisione di regime: **(a) intraday-only, flat 16:00** (raccomandata)
      ✔ **griglia 3m** e non 1m: a 1m breadth 18,94 e cross-section completa solo nel 33,1%
      ✔ tetto di attivazione **1 ora**; **N = 15** fissato (45 min), **W = 15 provvisorio** → M6
      ✔ griglia W per la M6 ridotta a **{10, 12, 15}**: a W=20 il 23% dell'etichetta è l'ora del giorno
      ✔ **taglio delle mezze giornate**: il nastro stampa fino alle 15:21 dopo la campana dell'una;
        48 barre su 118 non erano mercato. Le 21 mezze giornate tornano a 70 barre esatte
      ✔ 85 righe usabili su 130 (65,4%), verificato sullo store reale
      ✔ **test per troncamento**: si tronca a una chiusura, nessuna colonna della sessione
        successiva cambia
      ✔ test di anticipazione invariato: alle 10:05 il ramo 15m vede la barra chiusa alle 10:00
      ✔ nessun ffill e nessuna rolling attraversa una chiusura
- [ ] **Checkpoint B** — store riproducibile, M4 testata. Senza M4 ogni misura dopo è sospetta

## Fase C — il go/no-go

- [x] **M5** ✅ **2026-09-19** — `exhaustcheck` + libro contro `rotation_null`, **nessun training**
      → M2, M4
      ✔ controllo a **esposizione fissa** (`rotation_null`), mai buy-and-hold
      ✔ **lordo +0,5601 contro null −0,0001 ± 0,0198, z = 28,2** — lo sbarramento passa
      ✔ simmetria: segno invertito dà −0,5472, specchio quasi esatto
      ⚠ **netto −2,0851**: pareggio a 0,201 bp contro 0,281 del nome più economico di U1
      ⚠ serve **un quinto del turnover** a parità di lordo — da 10,4 a 2,2 rotazioni/sessione
- [x] **Checkpoint C** ✅ **go** — il lordo esiste e batte il null; il problema è raccoglierlo
- [x] **M5b** ✅ **2026-09-19** — finestra e frequenza: su quale scala il segnale paga
      ✘ **frequenza: non è una leva, è dannosa.** Stride 1→30: lordo ×0,030, turnover ×0,093
      ✔ **finestra: leva reale ma piccola.** Ottimo interno a w=30 (+21% sul rapporto), piatto a 40
      ⚠ l'ottimo (w=30 = 90 min, fondo 180) **sfora il tetto di attivazione di 1 ora** che cappa w a 20
      ⚠ miglior pareggio 0,242 bp contro 0,281 del nome più economico: ancora sotto
- [ ] **M5c** le due leve non provate: **selettività** (soglia sulla magnitudine, non isteresi) e
      **larghezza del libro** θ. Servono ~3,9× sul rapporto lordo/costo, la finestra ne ha dati 1,21

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

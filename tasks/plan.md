# Piano — dalla specifica al primo modello

Derivato da `equity_dataset_schema.html` §10, che è già un ordine di misure: **il più economico che
può uccidere il progetto, per primo**. Questo documento non lo riscrive, lo rende eseguibile su un
repo che oggi contiene due moduli (`data.candles`, `app.dashboard`) e nessuno di ricerca.

## Il principio che ordina tutto

> «Nessuna riga di training precede la misura 3. Il progetto precedente ha speso ore di GPU per
> ottimizzare una quantità che una misura da cinque minuti avrebbe squalificato.»

Tre fasi possono chiudere il progetto **prima** che venga addestrato qualcosa (M0, M5, e la
dispersione cross-sezionale). Sono messe davanti apposta. Un piano che le rimanda per "arrivare
prima al modello" ha invertito il senso del documento.

## Correzione alla §3, verificata il 2026-09-19

Lo storico **SIP è accessibile con le chiavi paper gratuite**, dal 2016-01-04, `adjustment=all`.
Misurato: SPY 1m del 2024-06-03 14:30 UTC legge `v=75.038 / n=2495` su SIP contro `v=577 / n=14`
su IEX — IEX è lo 0,8% di quella barra. A pagamento è solo il real-time ($99/mese), che serve in
live e non in backtest. **Conseguenza: `ALPACA_FEED` deve avere default `sip`, non `iex`**, e M3
non ha alcun blocco commerciale.

## Grafo delle dipendenze

```
M0 (chiude legsweep, repo TradingVision)   ── indipendente da tutto

M1a (esposizioni scorrelate → E1) ─→ M1b (spread → strumento per esposizione → U1) ─┐
                                                                                    ├─→ M2 (costs)
M3 (store 1m) ─→ M3.5 (dispersione intraday) ─→ M5 (kill test) ←────────────────────┘
      └─────────→ M4 (tempo di sessione) ──────────↑
                                                   │
M6 (oracle, W) ─→ M7 (linear, leakage) ─→ M8 (gbm, selection) ─→ M9 (swing) ─→ M10 (paper)
```

**M1a viene prima di M1b, e il costo non entra in M1a.** L'universo esiste per dare al modello
strutture scorrelate su cui imparare il mercato; il costo decide *quale strumento* implementa
un'esposizione già scelta, e deve essere corretto **nelle simulazioni**. Invertire i due significa
scegliere cosa il modello può imparare in base a quanto costa muoverlo, che è una scelta fatta col
criterio sbagliato.

Il codice di M2 si può scrivere in parallelo a M1a/M1b; i **numeri** per simbolo richiedono M1b.
Tutto il resto è una catena. **M4 sta sotto ogni misura successiva**: se arriva dopo M5, M5 va
rifatta.

## Il porting non è una fase

I moduli che §10 marca *riusati senza modifiche* (`split`, `metrics`, `normalize`, `data.pivots`,
`legs`, `swing.fit_policy`, `swingrule.rotation_null`, `legcheck`, `threshold`, `stops`, `oracle`)
si copiano **quando la prima misura che li usa li chiede**, non in un blocco iniziale. Un port
in blocco produce venti moduli non esercitati e nessuna misura; un port per slice arriva con il
suo test. `data.binance` non si porta: è fuori dal progetto.

---

## Fase A — le tre misure che non richiedono lo store

### M0 · Chiudere `legsweep` con un `legcheck` sulla cella argmax
**Dove:** repo `TradingVision`, non qui. **Costo:** 5 min.
**Decide:** se la griglia crypto avesse un vincitore, prima di abbandonarla.
**Fatto quando:** il decile «compra» e il decile «vendi» della cella (0,2 / 12) sono quotati contro
il prezzo con il loro |t|, e il risultato è scritto nella §5 dello schema.
**Chiude in negativo:** decile piatto → la griglia non aveva un vincitore, come `edge` già diceva
(0,0330 contro 0,0298). Nessuna conseguenza su questo repo: si smette di guardarla, non si cambia.

### M1a · Le esposizioni — **il set di dati, deciso senza guardare il costo**
**Costo:** 3 h. **Precede M1b.** Richiede storico **daily** dei candidati (una chiamata per simbolo,
gratuita), non lo store intraday.

Questa è la decisione sul **dataset**, ed è una decisione su cosa il modello può imparare. L'universo
serve a dare venti **strutture scorrelate**: l'etichetta è un ordinamento dentro l'istante, e venti
cloni dell'S&P hanno cross-section degenere — rumore diviso per una dispersione quasi nulla. Il
costo non entra qui. Entra in M1b, per scegliere *con quale strumento* si compra un'esposizione già
decisa, e in M2 perché le simulazioni siano oneste.

Questo **ordina un conflitto che la §4 lascia aperto**. Il documento dice che i criteri 1 (prezzo
alto) e 5 (dispersione reale) sono in conflitto e che la leva è l'esistenza di due emittenti per la
stessa esposizione. La priorità è: **criterio 5 sceglie le esposizioni, criterio 1 sceglie
l'emittente dentro l'esposizione.**

**Fatto quando:** esiste `E1`, venti esposizioni con la matrice di correlazione dei loro rendimenti
e una motivazione per ciascuna. Cardinalità **venti**: cambiarla ritara ogni soglia di
significatività (§8).
**Attenzione, regola del progetto:** *tutto ciò che decide qualcosa è misurato solo sul train* (§8).
Scegliere le esposizioni massimizzando la scorrelazione su **tutto** lo storico è una scelta presa
sulla fetta di test. La matrice va stimata sul primo fold e **verificata**, non rifatta, sugli altri.
**Non può fallire, può solo decidere.**

### M1b · Lo strumento per ogni esposizione — spread effettivo → U1
**Costo:** 3 h. **Nuovo modulo:** `data.quotes` + uno script di misura. → M1a
**Regola di decisione, fissata prima di guardare l'output** (§4): dentro ogni esposizione di E1
vince l'emittente con il costo per lato più basso secondo la formula §2, con **spread mediano
ponderato per il tempo dentro RTH, escluse la prima e l'ultima mezz'ora**.
**Fatto quando:** tabella dei candidati con spread mediano in centesimi e costo in bp, e `SYMBOLS`
contiene i venti strumenti scelti, ciascuno col commento che dice **quale esposizione implementa** e
da quale misura esce.
**Nota:** se dentro un'esposizione nessuno strumento è accettabile per costo, si cambia **strumento**
o si dichiara l'esposizione non tradabile — non si sostituisce l'esposizione con una più economica
ma correlata a un'altra già presente, che rimetterebbe il costo a decidere il dataset.

### M2 · `costs.py` — fee per simbolo in tutta la catena
**Costo:** 3 h. **Nuovo modulo:** `costs`. → M1b per i numeri, il codice si scrive prima.
**Il suo posto è la simulazione, non il dataset.** Nessuna scelta su quali dati il modello vede
passa da qui; ogni figura di P&L sì.
Implementa `costo_per_lato(bp) = 0,103 + 0,975/P + 5000·s/P` con SEC §31 e TAF **in config, mai
costanti** (la SEC cambia ogni anno; era $0,00/M fino al 3 apr 2026). Il dividendo su short entra
qui, per simbolo e **solo sulla gamba corta**, non nella serie.
**Cambio di tipo, non di architettura:** `--fee` è già argomento di `swing`, `threshold`, `stops`,
`swingrule`, `oracle`; diventa un vettore per simbolo. Chi lo consuma non cambia.
**Self-check obbligatorio:** con fee piatta i numeri del progetto precedente devono tornare
**identici**. È l'unico modo di sapere che il vettore non ha cambiato l'aritmetica.

> **Checkpoint A** — E1 (le esposizioni) e U1 (gli strumenti) sono definiti e distinti, il costo è
> per simbolo ed è confinato alle simulazioni, M0 ha un verdetto scritto.
> Niente store intraday, niente training, nessuna riga di modello.

---

## Fase B — lo store e il tempo di sessione

### M3 · Store Alpaca 1m SIP
**Costo:** 4 h. **Nuovo modulo:** `data.alpaca_equities` (bulk, un file per (simbolo, intervallo));
`data.candles` resta la lettura live della pagina.
Parametri non negoziabili (§3): `feed=sip`, `adjustment=all`, base **1Min** (5m e 15m si
ricampionano, il contrario no), RTH filtrata esplicitamente, dal 2016-01-04.
**Le tre trappole da controllare per prime, perché rompono in silenzio:**
1. fuso del timestamp restituito (UTC vs ET) attraverso i **due cambi DST** annuali;
2. comportamento sulle **mezze giornate** (24 dic, 29 nov, …);
3. se `adjustment=all` **riscrive retroattivamente** le barre a ogni nuovo split — se sì lo store
   non è riproducibile e lo stamp della cache non se ne accorge.
**Fatto quando:** i venti file esistono, e le tre trappole hanno una risposta misurata scritta nel
docstring del modulo. **Stamp della cache** che rifiuta parametri diversi (§8, `dataset.cached`).
**Nuovo modulo:** `calendar` (NYSE: festivi, mezze giornate, DST).

### M3.5 · Dispersione cross-sezionale — **da fare subito dopo M3, prima della M5**
**Costo:** 1 h. Non ha numero in §10 perché è una delle «due incognite che nessun backtest vede»,
ma il documento dice esplicitamente *quando*: subito dopo la 3.
**Perché:** l'etichetta è un ordinamento dentro l'istante **diviso per la dispersione dell'istante**.
Venti ETF settoriali USA condividono un fattore di mercato fortissimo, a frequenza intraday forse
più di venti altcoin. Se `sd_t` è piccola, l'etichetta è rumore diviso per un numero piccolo.
**Non è un doppione di M1a.** M1a è *progetto* e si misura su rendimenti daily: queste esposizioni
si muovono davvero in modo diverso? M3.5 è *validazione* alla frequenza su cui si opera: a 1-15
minuti il fattore di mercato comune domina molto più che su daily, quindi un E1 ben scelto **può
comunque** fallire qui. Serve lo store intraday e non si può anticipare.
**Fatto quando:** esiste la distribuzione di `sd_t` per data su U1, a 1m/5m/15m, confrontata con la
stessa quantità misurata su venti coppie crypto.
**Chiude in negativo:** `sd_t` di un ordine sotto il crypto → l'etichetta cross-sezionale non ha
materia prima, e va ripensata prima di costruirci sopra.

### M4 · Tempo di sessione — **il refactor principale del progetto**
**Costo:** 1 giorno. Tocca `dataset.build` e la costruzione della griglia in `data.candles`.
La regola vecchia («fra due barre adiacenti passa sempre una durata di barra») assume tempo
continuo. Fra le 15:55 e le 09:30 passano 17,5 ore, nel weekend 65. **Cinque rotture concrete:**
ffill che attraversa la notte; `index.floor(step) == index` che genera timestamp alle 03:00;
ATR/rolling che leggono il gap d'apertura come range di barra; `find_pivots` con `order=W` che
unisce un massimo di ieri e uno di stamattina nella stessa oscillazione; `complete()` che
riempirebbe la notte con 210 barre finte.
**La correzione:** **indice di barra locale alla sessione** — sessioni concatenate, intero
progressivo che non salta mai, timestamp degradato a colonna (serve solo a split temporale, join
fra rami, calendario societario).
**Perché è più piccola di quanto sembri:** tutto a valle è già **posizionale** — `features` su N
barre, `find_pivots` su `order` barre, il tensore su 24 step, il purging su indici di riga.
**Regola derivata, da mettere a test:** *nessuna finestra rolling e nessun forward fill attraversa
una chiusura.* **Verifica:** lo stesso **test per troncamento** che già protegge l'allineamento — si
tronca la serie a una chiusura e si verifica che **nessuna colonna della sessione successiva
cambi**. Più il test di anticipazione invariato (alle 10:05 il ramo 15m vede la barra chiusa alle
10:00, non quella in formazione).
**Decisione di regime, raccomandata dal documento:** **(a) intraday-only, flat alle 16:00.** Toglie
il gap da etichetta *e* da rischio; il drift overnight è esattamente l'esposizione che ha ingannato
il progetto precedente; la PDT è ritirata dal 4 giu 2026 e il minimo 4x è $2.000. Costo dichiarato:
si rinuncia al drift, che per un libro market-neutral è una rinuncia piccola.
**Conseguenza su §6:** tre rami (1m, 5m, 15m) e non quattro, griglia base 1m — a N=24 il 15m copre
6h, cioè una sessione, che è **il tetto e non un valore**. 30m e 1h attraversano una chiusura.

> **Checkpoint B** — lo store è riproducibile, il tempo di sessione è testato per troncamento, la
> dispersione ha un numero. **Se M4 non è chiusa, ogni misura successiva è sospetta.**

---

## Fase C — il test che può chiudere il progetto in due giorni

### M5 · `exhaustcheck --price` e baseline di `factor` su U1, col costo vero
**Costo:** 1 h. **Nessun training.** Porta `exhaustcheck`, `factor`, `metrics`, `split`,
`swingrule.rotation_null`.
**Decide: se il progetto ha senso.**
**Chiude in negativo:** lordo ≤ 0 contro `rotation_null` → **nessun lavoro di modello lo salva**, e
sono stati spesi due giorni invece di due settimane.
**Verifica:** il controllo è a **esposizione fissa**, mai il buy-and-hold — su un periodo in salita
il buy-and-hold farebbe sembrare bravura lo stare lunghi, che è l'errore speculare a quello crypto.

> **Checkpoint C — go/no-go del progetto.** Qui si decide se continuare.

---

## Fase D — etichetta, leakage, riferimento

### M6 · `oracle` — le due letture e `finestra_estremi`
**Costo:** 2 h. `lag=0` (hindsight) e `lag=W` (il primo istante conoscibile). Si quota sempre il
secondo; il primo si cita solo per dire cos'è.
**Da rimisurare:** W, col metodo crypto **integralmente riusato** (P&L d'oracolo penalizzato dal
ritardo; il criterio ingenuo è degenere, l'argmax cade a 3 barre e non è stazionario).
**Vincolo nuovo:** a `L = finestra` la gamba deve stare **dentro la sessione**. Su 5m W=24 sono 2h
su 6,5 — plausibile; su 15m sono 6h, cioè il tetto.
**Rimisurare anche:** orizzonte di purging in **barre di sessione** (crypto: p50 28 / p95 124 /
p99 202 / **max 754**) — il massimo è il numero che conta, perché il purging è esatto.

### M7 · `linear` — allarme leakage
**Costo:** 1 h. Attesa: **Rank IC ≈ 0** sull'etichetta predittiva.
**Chiude in negativo:** Rank IC alta → **c'è leakage di sessione, si torna alla M4.** È il test che
dice se il refactor più costoso è stato fatto bene.

### M8 · `gbm --horizon`, poi `selection` rifatta su U1
**Costo:** 4 h. La selezione 28→12 crypto è **decaduta**, non si trasferisce.
**Regola:** il set è un parametro (`--features`), non una cancellazione — per la GRU le 28 complete
battevano il set ridotto su **quattro fold su quattro**.
**Aperto, da decidere con una misura e non a priori** (§6): la stagionalità intraday a U del volume
domina sei colonne. Due rimedi da **confrontare**: (i) normalizzare ogni colonna di volume contro la
mediana storica della stessa fascia oraria; (ii) lasciarle grezze e aggiungere seno/coseno della
frazione di sessione. La (ii) è meno codice e più onesta; la (i) rende le colonne confrontabili fra
simboli, che è ciò che serve a un'etichetta cross-sezionale.
**Cautela obbligatoria:** l'ora del giorno è la variabile più prevedibile del mercato. Serve una
**baseline a sola ora del giorno**, esattamente come `rsi_centered` controlla `legsweep`.

---

## Fase E — il modello, e solo qui

### M9 · `swing` due stadi, quattro fold, cinque seed
**Costo:** 1 giorno. Stadio 1 Huber su `swing_leg_target` (mette la struttura di gamba
nell'encoder); stadio 2 `fit_policy` sul **P&L netto**, fee per simbolo dentro la ricompensa,
rendimenti **detrendizzati**. δ della Huber **rimisurato** — 2,1 non ha significato qui.
**Promozione: il criterio di §9, in quest'ordine, ogni riga è uno sbarramento:**
1. `legcheck` contro il **rendimento forward**: decile compra > 0, decile vendi < 0, entrambi
   |t| > 2, monotonia fra i due. **Senza questa riga non c'è niente su cui una regola possa agire.**
2. Lordo sopra `rotation_null`, ≥ 500 rotazioni, **z > 2**. (Sul venue precedente: p = 0,33.)
3. Netto positivo **col costo per simbolo**, non con una media.
4. **Information ratio**, non log per anno. Con 4x a partire da $2.000 il rendimento è una leva; lo
   Sharpe no.
5. Dispersione sui **quattro fold**, che ora sono quattro regimi.

**Mai:** promuovere sulla Rank IC contro l'etichetta. La predizione che faceva **0,4114** contro
l'etichetta faceva **−0,0405** contro il prezzo. Un miglioramento di Rank IC che non muove
`legcheck` non è un miglioramento.

### M10 · Paper trading
**Costo:** 1 settimana. Verifica **locate sugli ETB** e **slippage reale contro il mid**.
**Il rischio più grande del progetto, verificabile il primo giorno:** se il locate serve anche sugli
ETB, una regola always-in che flippa ~200 volte l'anno per simbolo fa 200 locate per simbolo, con
latenza e possibilità di fallire. **Se cade:** U1 resta valido ma il libro diventa long-only più
hedge di indice — **una strategia diversa, da prezzare come tale.**

---

## Cosa non rifare (§9, lista ereditata)

- Leggere un **win rate** come un edge. 0,545 ± 0,013 su 1577 trade è tre sigma sopra la moneta e
  non vale niente, perché il trade medio era negativo.
- Confrontare una regola long-only col **solo buy-and-hold**.
- Cercare la colpa **nella regola di trading**: `legcheck` è senza regola e senza commissioni. Se è
  piatta lì, la regola non è il problema.
- Leggere un hit rate per dimensione del movimento **senza la colonna delle durate** accanto.
- Fidarsi di un `pred-*.parquet` **senza controllarne la breadth**:
  `d.groupby(d.index.get_level_values(0)).size().mean()`. Se non è ~20, ogni metrica
  cross-sezionale su quel file è diluita.
- Migliorare la Rank IC contro l'etichetta e chiamarlo progresso.

## Convenzioni per ogni task

Ogni slice porta con sé: self-check in fondo al modulo, **iscrizione in `tests/test_selfchecks.py`**,
docstring che dice cosa è stato misurato e cosa è uscito, `ruff`/`black`/`pytest` verdi, e
l'aggiornamento della sezione corrispondente di `equity_dataset_schema.html`. Il soggetto del commit
è la misura, non il file: *«Quattro rami perdono contro uno, su ogni fold»*.

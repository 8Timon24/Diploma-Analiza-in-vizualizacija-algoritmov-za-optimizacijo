# Analiza in vizualizacija algoritmov za optimizacijo prek iskalnih trajektorij

**Koda diplomske naloge**

*(This README is also available in English: [README.md](README.md))*

Metahevristične optimizacijske algoritme običajno primerjamo po tem, *kako dobra* je
njihova končna rešitev. Ta projekt jih primerja po tem, *kako iščejo*: zabeleži celotno
trajektorijo populacije za 28 metahevrističnih algoritmov na naboru testnih problemov BBOB,
vsako trajektorijo diskretizira s clusteringom obiskanih točk in izpelje več paričnih mer
"vedenjske razdalje" med algoritmi.

Raziskovalno vprašanje je, katere od teh mer dejansko povedo kaj različnega. Zadnji korak
izračuna Spearmanovo korelacijo med vsemi metrikami in loči tiste, ki so večinoma redundantne,
od tistih, ki nosijo komplementarno informacijo.

## Vsebina

- [Kaj se meri](#kaj-se-meri)
- [Dva načina uporabe](#dva-načina-uporabe)
- [Zahteve](#zahteve)
- [Cevovod prek ukazne vrstice](#cevovod-prek-ukazne-vrstice)
- [Faze cevovoda](#faze-cevovoda)
- [Izhodi](#izhodi)
- [Regresija nad skalarji po algoritmih](#regresija-nad-skalarji-po-algoritmih)
- [Raziskovalni zvezki](#raziskovalni-zvezki)
- [Namizna aplikacija](#namizna-aplikacija)
- [Testi](#testi)
- [Opomba o ponovljivosti](#opomba-o-ponovljivosti)

## Kaj se meri

Vsaka metrika vrne eno vrednost za vsako kombinacijo (algoritem A, algoritem B, funkcija,
instanca, ponovitev):

| Metrika | Kaj meri |
|---|---|
| `entropy` | Razlika v normalizirani Shannonovi entropiji (H / ln k) zasedenosti gruč — kako razpršena je populacija |
| `cosine` | Kosinusna razdalja med sploščenima vektorjema zasedenosti gruč |
| `cosine_columns` | Kosinusna razdalja, izračunana po posamezni gruči, nato povprečena |
| `exploration` | Razlika v razmerju med raziskovanjem (exploration) in izkoriščanjem (exploitation) |
| `location` | Evklidska razdalja med dvema končnima rešitvama |
| `fitness` | Razlika v končni vrednosti kriterijske funkcije |

## Dva načina uporabe

Ena koda, dva vmesnika nad njo. Namizna aplikacija **uvozi** (import) cevovod —
`config.py`, `helper_functions.py`, in funkcije za risanje v `04_metrics/` in
`05_analysis/` — namesto da bi karkoli od tega podvojila, zato je slika, narisana
v aplikaciji, ista slika, ki jo ustvari cevovod.

| | Kaj je | Kdaj poseči po njem |
|---|---|---|
| **[Cevovod prek ukazne vrstice](#cevovod-prek-ukazne-vrstice)** | Delovni tok diplomske naloge: 14 paketnih korakov od meritve do Spearmanove korelacije | Za reprodukcijo rezultatov naloge ali zagon celotnega nabora meritev |
| **[Namizna aplikacija](#namizna-aplikacija)** | Grafični vmesnik v PySide6: izbira algoritmov, zagon omejenih eksperimentov, brskanje po podatkih, izris slik | Za raziskovanje rezultatov, preizkušanje drugih algoritmov ali naborov testnih problemov, predstavitev dela drugim |

Cevovod je izvor resnice in deluje samostojno; aplikacija je opcijska in nič v
cevovodu je ne uvaža.

## Zahteve

Python 3.10 in paketi iz `requirements.txt`. Tisti, ki lahko zahteva dodatno pozornost, je
`cocoex` (`coco-experiment`), nabor testnih problemov COCO/BBOB — gre za prevedeno
razširitev v C.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Nobenih map ni treba ustvariti ročno. `data/`, `outputs/`, `metrics_data/` in mape
`figures_*/` so generirane, izključene iz git (gitignored) in jih na svežem kloniranju
ni; vsak korak ob prvem pisanju ustvari, kar potrebuje.

## Cevovod prek ukazne vrstice

`run_pipeline.py` zažene vseh 14 korakov po vrsti, vsakega kot ločen podproces, in se
ustavi ob prvi napaki, da se nič ne izvede na pokvarjenih podatkih:

```bash
python run_pipeline.py                     # vse, od začetka do konca
python run_pipeline.py --dry-run           # izpiše ukaze, ne da bi jih zagnal
python run_pipeline.py --from clustering   # nadaljuje od izbranega koraka (vključno z njim)
python run_pipeline.py --only spearman     # zažene samo en korak
```

**Prvi korak je potraten.** Zažene 28 algoritmov × 24 funkcij × 5 instanc ×
3 dimenzije × 5 semen (seeds), pri čemer je `epoch = 10 × dimenzija` in velikost
populacije 50. Računajte na ure, ne minute. Vse za tem deluje na predpomnjenih
rezultatih, zato v praksi meritev zaženete enkrat, kasnejše korake pa ponovno
poganjate z `--from`.

Posamezne korake lahko zaženete tudi neposredno; delujejo iz katerekoli delovne mape:

```bash
python 01_optimize/run_benchmarks.py -a OriginalGWO -f 1 -d 2 -i 1 -s 1   # majhen podnabor
python 03_cluster/cluster_trajectories.py -c kmeans
```

### Vsak korak, zagnan neposredno

Natančen ukaz, ki ga `run_pipeline.py` zažene za vsak korak (iz `STEPS` v
`run_pipeline.py`), po vrsti:

| Korak | Ukaz | Ustvari |
|---|---|---|
| `benchmark` | `python 01_optimize/run_benchmarks.py -p a -e` | surove trajektorije posameznih zagonov v `outputs/` |
| `harvest_results` | `python 01_optimize/harvest_results.py` | `outputs/dim_{d}/results.csv` |
| `preprocess` | `python 02_preprocess/preprocess_data.py` | `data/processed/dim_{d}/F{f}_I{i}.csv` |
| `clustering` | `python 03_cluster/cluster_trajectories.py -c kmeans` | `data/clustering_latest/{cluster_centers,cluster_distributions,clustering_results}` |
| `aggregate_cosine` | `python 03_cluster/cluster_similarity.py -c kmeans` | agregirana kosinusna podobnost, za slike tipa clustermap |
| `entropy_calc` | `python 04_metrics/entropy.py` | entropija zasedenosti gruč (podrobna + agregirana) |
| `entropy_pairwise` | `python 04_metrics/entropy_pairwise.py` | parična metrika razlike v entropiji |
| `cosine_pairwise` | `python 04_metrics/cosine_pairwise.py` | parična globalna kosinusna razdalja |
| `cosine_columns_pairwise` | `python 04_metrics/cosine_columns_pairwise.py` | parična kosinusna razdalja po posameznih gručah |
| `exploration_pairwise` | `python 04_metrics/exploration_pairwise.py` | parična razlika v razmerju raziskovanje/izkoriščanje |
| `solutions_pairwise` | `python 04_metrics/solutions_pairwise.py` | parična razlika v lokaciji/vrednosti rešitve |
| `merge` | `python 05_analysis/merge_metrics.py` | `metrics_data/merged/merged_dim_{d}.csv` |
| `build_scalars` | `python 05_analysis/build_scalars.py` | `metrics_data/scalars.csv` |
| `spearman` | `python 05_analysis/spearman.py` | `metrics_data/merged/spearman_dim_{d}.csv`, `figures_spearman/` |

Dve dodatni skripti nista koraka `run_pipeline.py` - zaženite ju ročno, potem ko je
njuna odvisnost že izvedena:

| Ukaz | Potrebuje | Ustvari |
|---|---|---|
| `python 04_metrics/entropy_plotting.py` | `entropy_calc` | slike entropije v `figures_entropy/` |
| `python 05_analysis/scalar_regression.py` | `build_scalars` | `figures_results/scalar_regression.png` (glej [spodaj](#regresija-nad-skalarji-po-algoritmih)) |

## Faze cevovoda

Skripte so v oštevilčenih mapah, ki sledijo vrstnemu redu izvajanja:

```
01_optimize/    zagon algoritmov na BBOB, zbiranje najboljših rezultatov
02_preprocess/  preoblikovanje surovih trajektorij v vhodni format za clustering
03_cluster/     KMeans clustering točk trajektorij; agregacija kosinusne podobnosti
04_metrics/     entropija, kosinus, kosinus po stolpcih, raziskovanje, lokacija/vrednost rešitve
05_analysis/    združevanje vseh metrik, Spearmanova korelacija, raziskovalni zvezki
```

Podporna koda je v korenu repozitorija: `config.py` (vse konstante in poti),
`utils.py` in `helper_functions.py` (skupne pomožne funkcije), `run_pipeline.py`
(orkestrator). `scratch/` vsebuje enkratne diagnostične skripte, `tests/` pa testni nabor.

`gui/` je namizna aplikacija, `packaging/` pa njena specifikacija za gradnjo. Oboje
je poleg cevovoda, ne pa ovoj okoli njega: oštevilčene faze zgoraj delujejo natanko
tako kot vedno, ne glede na to, ali je aplikacija nameščena.

`config.py` je edino mesto za spremembo seznama algoritmov, nabora dimenzij/funkcij/
instanc/semen, semena za clustering in vseh vhodnih/izhodnih map.

## Izhodi

```
outputs/        surove trajektorije posameznih zagonov algoritmov
data/           obdelane trajektorije, rezultati clusteringa, tabele entropije
metrics_data/   ena CSV datoteka na metriko na problem, plus merged_dim_{d}.csv
figures_*/      generirane slike
```

`metrics_data/merged/merged_dim_{d}.csv` je glavni izdelek: vse metrike, združene
prek (Algorithm1, Algorithm2, Function_id, Instance_id, Run_id).

`metrics_data/scalars.csv` je spremljajoča tabela po **algoritmih** (ena vrstica na
dimenzijo/funkcijo/algoritem, z entropijo, vrednostjo rešitve, raziskovanjem in
raznolikostjo), uporabljena za spodnjo regresijsko analizo.

## Regresija nad skalarji po algoritmih

Zgornje parične metrike odgovorijo na vprašanje "kako različno iščeta A in B?". Za
vprašanje o samih algoritmih — *ali algoritmi z višjo entropijo dejansko bolj raziskujejo?*
— potrebujete skalarje po posameznem algoritmu namesto paričnih razlik, kar zagotavlja
`metrics_data/scalars.csv`:

```bash
python 05_analysis/build_scalars.py                      # zgradi tabelo skalarjev
python 05_analysis/scalar_regression.py                  # entropija proti raziskovanju (privzeto)
python 05_analysis/scalar_regression.py --x entropy --y fitness --dims 2 10
```

Skripta prilagodi en OLS panel na dimenzijo, obarva 28 algoritmov po družinah, nariše
merilne črtice za razpršenost po BBOB funkcijah in obkroži vplivne točke (Cookova
razdalja), tako da je premica, ki jo podpirata le en ali dva osamelca, vidna in ne
skrita. Izhod gre v `figures_results/scalar_regression.png`.

Pomembno: namenoma regresira *skalarje*, ne povprečenih paričnih razdalj: pri metriki
tipa razlike je povprečenje čez partnerje V-oblike v osnovnem skalarju, zato meri
netipičnost namesto velikosti (glej dokumentacijski niz modula).

## Raziskovalni zvezki

`05_analysis/` vsebuje tudi zvezke (notebooks), uporabljene za gradnjo in preverjanje
zgornje analize - `run_pipeline.py` jih ne zažene in nič v nadaljevanju ni odvisno od
njihovega izhoda. Odprite jih z `jupyter lab 05_analysis/` (ali prek podpore za zvezke
v urejevalniku) iz katerekoli mape; prva celica vsakega je **"bootstrap korenske mape"**,
ki z `os.chdir` skoči v koren repozitorija in ga doda na `sys.path`, saj je Jupyterjeva
lastna delovna mapa mapa zvezka, vse poti v teh zvezkih (`data/...`, `metrics_data/...`)
pa predpostavljajo koren repozitorija. To celico vedno zaženite prvo.

| Zvezek | Za kaj je namenjen |
|---|---|
| `3_cluster_analysis.ipynb` | Izris surovih trajektorij populacije za izbran nabor dimenzij/problemov/algoritmov/ponovitev, obarvanih po gruči |
| `4_example_visualize_trajectories_for_clustering.ipynb` | Delujoč primer vizualizacije krajine problema in trajektorij, od priprave vhoda za clustering naprej |
| `entropy_notebook.ipynb` | En primer vsake vrste slike entropije (`04_metrics/entropy_plotting.py`), s filtriranjem algoritmov |
| `exploration_plots_preview.ipynb` | Krivulje raziskovanja/izkoriščanja, primerjane med algoritmi |
| `location_fitness_preview.ipynb` | Lokacija in vrednost končne rešitve, primerjani z resničnim optimumom BBOB |
| `pari_algoritmov_metrike_heatmap.ipynb` | Toplotna karta in tabela paričnih metrik za hitro oceno, katere mere se ujemajo, pred avtomatiziranim korakom Spearman |

Zvezki so veliki (eden ima nekaj MB vgrajenega izhoda slik) - če enega urejate
programsko namesto prek Jupyterja, ga naložite z `json.load`/`json.dump(...,
indent=1, ensure_ascii=False)` plus zaključnim novim vrstico, namesto z urejevalnikom
besedila, da diff ostane omejen samo na vašo spremembo.

## Namizna aplikacija

Grafični vmesnik v PySide6 nad istim cevovodom: izbira algoritmov in testnih funkcij,
zagon omejenega eksperimenta, brskanje po surovih podatkih in izris slik projekta,
brez pisanja Pythona.

```bash
python -m gui                 # iz korena repozitorija, v venv
python -m gui --self-test     # brezokenski (headless) preizkusi, brez okna; uporablja ga CI za gradnjo
```

Sedem zavihkov:

| Zavihek | Kaj počne |
| --- | --- |
| Setup | Izbira med vsemi 234 algoritmi mealpy in virom testnih problemov, z oceno stroška v živo |
| Run | Napredek, graf konvergence v živo, sprotni dnevnik, preklic |
| Process | Zažene clustering in vsako fazo metrik - preostanek cevovoda po meritvi - brez terminala |
| Visualize | 20 vizualizacij, 13 od njih prej dosegljivih samo z zagonom zvezka |
| Trajectory | Animirano 2D predvajanje iskanja nad resnično krajino, z izvozom v GIF |
| Compare | Dva algoritma vzporedno, prek vseh šestih paričnih metrik |
| Data | Vsaka datoteka, ki jo je cevovod zapisal, v razvrstljivi tabeli |

Viri testnih problemov: **BBOB** (24 funkcij x 110 instanc prek COCO), **opfunu**
(125 klasičnih funkcij), nabori **CEC** (cec2005-cec2022) in **uporabniško
definiran** izraz.

### Aplikacija po nesreči ne spreminja vaših rezultatov

Zagoni meritev (zavihek Run) se zapišejo v `gui_runs/<časovni žig>/` z datoteko
`manifest.json`, ki zabeleži natančno izbiro, verzije knjižnic in seme za clustering
- nikoli v pravi `outputs/` repozitorija, razen če to izrecno potrdite v zavihku Setup.
Izris slike nikoli ne piše v katerokoli mapo `figures_*/` - funkcije cevovoda za risanje
kličejo `savefig` s trdo zapisanimi potmi, zato aplikacija med izrisom `savefig` onemogoči,
izvoz slike pa je ločeno, izrecno dejanje.

Zavihek Process je edino mesto, ki *dejansko* piše v prave mape `data/` in
`metrics_data/` na mestu samem - glej spodaj - in tega nikoli ne stori brez vaše
potrditve, kaj natančno bo prepisano.

Če vizualizacija potrebuje podatke, ki še ne obstajajo, aplikacija pove natanko, kaj
manjka, oceni, koliko bi to stalo, in ponudi, da te faze zažene namesto vas (ali, za
meritev, vnaprej izpolni obrazec Run s to izbiro).

### Zagon cevovoda iz aplikacije

Zavihek **Process** zažene vsako fazo po meritvi - predobdelavo, clustering, vseh
šest paričnih metrik, združevanje, skalarje, Spearman - znotraj tekoče aplikacije,
brez terminala. Označite posamezne faze ali uporabite "start from", da označite fazo
in vse za njo (enako, kot na ukazni vrstici počne `run_pipeline.py --from <korak>` -
oba se ujemata po imenih in vrstnem redu korakov). Dva prikaza napredka sledita
celotnemu napredku in napredku trenutne faze; Cancel se ustavi ob naslednji meji
posameznega elementa.

Piše v prave mape `data/` in `metrics_data/` na mestu samem, enako kot zagon korakov
iz terminala, zato klik na Run najprej zahteva potrditev - z navedbo vsake mape, ki
bo prepisana, in njene trenutne velikosti. Zagoni niso transakcijski in nič se ne
izbriše vnaprej, zato preklic na sredini pusti mešanico starih in novih datotek;
tako potrditveno okno kot dnevnik to povesta. Datoteka `pipeline_run.json` se zapiše
poleg rezultatov, z zapisom, katere faze so tekle, koliko časa in katero seme za
clustering je veljalo.

To je tudi tisto, kar naredi **paketno gradnjo** samostojno: ta ne vsebuje interpreterja
Python, zato pristop `run_pipeline.py` (podproces na korak) tam ni na voljo - zavihek
Process namesto tega uvozi vsak modul faze in ga pokliče znotraj istega procesa, zato
sploh obstaja.

### Usmerjanje aplikacije na vaše rezultate

**File > Open results folder...** preklopi, katero drevo aplikacija bere in vanj piše
(`outputs/`, `data/`, `metrics_data/`), izbira pa se zapomni. To je tisto, kar naredi
paketno aplikacijo uporabno na računalniku brez repozitorija; glava zavihka Data
vedno pove, katera mapa je aktivna in kaj vsebuje.

**File > Open a run from this app...** odpre enega od vaših lastnih zagonov. Mapa
zagona je drevo rezultatov sama zase, zato pogledi na surove trajektorije in
raziskovanje delujejo takoj; za poglede entropije, kosinusa in Spearman zaženite
zavihek Process nad njo.

### Pridobitev izvedljive datoteke za Windows

Potisnite oznako različice (`git tag v0.1.0 && git push origin v0.1.0`) in
`.github/workflows/build-windows.yml` jo zgradi ter zapakiran paket pripne k GitHub
Release pod to oznako - ta stran Release je tisto, kar posredujete nekomu, ki želi
samo zagnati aplikacijo, ne repozitorija. Prenesite zip, ga razpakirajte (`.exe`
potrebuje spremljajoče DLL-je in podatkovne datoteke poleg sebe - ne kopirajte ga
posebej ven) in zaženite `OptimizerTrajectoryExplorer.exe`. Ni digitalno podpisan,
zato ga bo Windows SmartScreen ob prvem zagonu označil; kliknite **More info > Run
anyway**.

Ročni zagon delovnega toka (zavihek Actions > build-windows-exe > Run workflow)
namesto prek oznake zgradi enako, a naloži samo kot 30-dnevni artefakt delovnega
toka, saj ni oznake, po kateri bi poimenovali Release.

PyInstaller ne zna navzkrižno prevajati, zato je `.exe` zgrajen izključno na
izvajalcu Windows - za gradnjo lokalno na svojem računalniku z Windows namesto tega:

```bash
pip install -r packaging/requirements-app.txt
pip install --no-deps mealpy==3.0.3   # razlog je v komentarju v tej datoteki
pyinstaller packaging/gui.spec --noconfirm
dist/OptimizerTrajectoryExplorer/OptimizerTrajectoryExplorer --self-test
```

Paket je velik približno 500 MB, samopreizkus (self-test) pa je tisto, kar dokaže,
da je uporaben: preveri stvari, ki odpovejo *šele* po pakiranju - optimizatorje mealpy
najde `pkgutil.walk_packages`, česar PyInstaller ne zna statično zaznati; opfunujevi
nabori CEC potrebujejo približno 1190 priloženih podatkovnih datotek; cocoex pa je
prevedena razširitev v C.

## Testi

```bash
pytest tests/                             # hitri enotski + GUI testi, ~10s, brez pravih podatkov
pytest tests/smoke_test_pipeline.py -v -s       # celoten cevovod prek podprocesov, s pravimi podatki
pytest tests/smoke_test_gui_pipeline.py -v -s   # enako, a znotraj istega procesa prek aplikacije
```

Oba dimna (smoke) testa zaženeta vseh 14 korakov v resnici (2 algoritma, 2 funkciji,
ena dimenzija) v izoliranem začasnem imeniku, zato nobeden nikoli ne dotakne vaših
pravih rezultatov. Prvi zažene vsak korak tako, kot to počne `run_pipeline.py`, kot
podproces; drugi enak nabor požene prek `gui/core/pipeline.py` - kode, ki jo uporablja
zavihek Process - vključno z enim testom, ki zažene pravi `QThread` in preizkusi sam
zavihek Process. Oba namenoma nista poimenovana `test_*.py`, zato ju `pytest tests/`
ne zazna samodejno in ju je treba zagnati izrecno.

## Opomba o ponovljivosti

Clustering je semenjen (`CLUSTERING_SEED` v `config.py`), zato so zagoni ponovljivi.
Rezultati, ustvarjeni *pred* uvedbo tega semena, niso — sklearnov KMeans naključno
inicializira centroide, na teh podatkih pa se dve različni semeni ne strinjata pri
približno polovici oznak gruč, kar premakne vsako metriko, izpeljano iz zasedenosti
gruč. Če potrebujete rezultate, ki jih ponoven zagon natančno ponovi, jih ponovno
generirajte od koraka clustering naprej.

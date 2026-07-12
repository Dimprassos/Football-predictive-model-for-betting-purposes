# PLAN.md — Αξιολόγηση κώδικα & σχέδιο εργασιών

> Το αρχείο αυτό φορτώνεται από το `AGENTS.md` ως context για κάθε coding agent.
> Περιέχει: (Α) τι είναι στέρεο και ΔΕΝ θέλει αλλαγή, (Β) τι θέλει αλλαγή/προσοχή,
> (Γ) το σχέδιο για την αναφορά (report) της διπλωματικής, που δεν έχει ξεκινήσει.
> Βάση αξιολόγησης: το `goal.txt` (στόχος + δομή αναφοράς). Τελευταία ενημέρωση: 2026-07-12.

## Συνολική εικόνα

Το σύστημα υλοποιεί σχεδόν πλήρως τον στόχο του `goal.txt`: εργαλείο πρόβλεψης 1/X/2
με επιλογή μοντέλου, επίπεδο εμπιστοσύνης, αξιολόγηση με πολλαπλές μετρικές,
επανεκπαίδευση με δεδομένα χρήστη, ensemble και βαθιά μάθηση. Ο πυρήνας
(pipeline, leakage control, αξιολόγηση) είναι σε πολύ καλή κατάσταση και **τα 76
unit tests περνούν** (`python -m unittest tests.test_core_behaviors`, 2026-07-12).
Οι εκκρεμότητες είναι κυρίως: 3–4 λειτουργικά κενά στο UI ως προς το goal.txt,
λίγα σημεία τεχνικού χρέους, και —το μεγαλύτερο— η **αναφορά, που δεν έχει γραφτεί**.

---

## Α. Τι είναι στέρεο (δεν θέλει αλλαγή)

1. **Leakage-safe χρονικό pipeline.** Σταθερά date-based splits
   (`src/config.py`: train < 2024-07, validation 2024-25, test ≥ 2025-07),
   streaming ανακατασκευή features με κανόνα `date < D`
   (`src/state_builder.py:streaming_block_probs_home_away`), opening odds ως
   pre-match αγορά, το τελικό σκορ δεν μπαίνει ποτέ στα features. Το ίδιο ισχύει
   και για τις ακολουθίες του FootyNet (`src/sequence_data.py:last_k_before`,
   με hard assertion ευθυγράμμισης στο `src/footynet_data.py`).

2. **Έντιμη μεθοδολογία επιλογής/αξιολόγησης.** Ό,τι «κερδίζει» επιλέγεται στο
   validation, ποτέ στο test: feature subsets, blend weights, market correction
   (`src/models/meta.py`), betting φίλτρα (`src/bet_selection.py` —
   validation-locked), σύσταση μοντέλου στοιχημάτων
   (`src/trainer.py:_select_recommended_betting_model`). Η αγορά αντιμετωπίζεται
   ρητά ως near-optimal benchmark και όχι ως στόχος «να νικηθεί» — σωστό frame
   για διπλωματική.

3. **Πλήρες φάσμα μοντέλων που ζητά το goal.txt.** Βάση Elo + Poisson/Dixon-Coles
   (`src/elo.py`, `src/poisson_model.py`), learned meta-models XGBoost/MLP/LogReg
   (`src/models/meta.py`), ensemble blend, βαθιά μάθηση FootyNet (LSTM/GRU
   late-fusion, `src/models/footynet.py`) και stacking FootyNet+αγορά
   (`src/footynet_stack.py`). Η σύνθεση αλγορίθμων (ensemble) και το deep learning
   —που το goal ζητούσε «να διερευνηθούν»— είναι υλοποιημένα και αξιολογημένα.

4. **Αξιολόγηση με πολλές μετρικές + βαθμονόμηση.** Log loss, Brier, top-label
   ECE, accuracy, macro-F1, per-class precision/recall (`src/metrics.py`),
   temperature scaling παντού (`src/calibration.py`), per-class marginal
   calibration για το draw πρόβλημα (`src/trainer.py:write_class_marginal_calibration`),
   bootstrap CIs (`scripts/analyze_ci.py`). Υπερκαλύπτει την απαίτηση του goal.

5. **Το Streamlit εργαλείο (`app.py`).** 4 σελίδες (Πρόβλεψη / Αξιολόγηση /
   Εκπαίδευση-Δεδομένα / Μεθοδολογία), δίγλωσσο (el/en) με πλήρεις πίνακες
   μετάφρασης, επίπεδο εμπιστοσύνης ανά πρόβλεψη, σύγκριση όλων των μοντέλων,
   προσθήκη αγώνων σε ξεχωριστό `zz_user_added.csv` (τα αρχικά δεδομένα δεν
   πειράζονται), retrain σε ξεχωριστό experiment `user_retrain` ώστε τα canonical
   νούμερα της εργασίας να μένουν άθικτα.

6. **Reproducibility / artifact discipline.** `PIPELINE_VERSION` + manifest +
   data fingerprint (`src/trainer.py:_cached_artifacts_are_compatible`),
   per-experiment paths ώστε πειράματα να μην αλληλογράφονται
   (`src/config.py`), deterministic seeds, γραμμένα per-match predictions για
   offline ανάλυση.

7. **Tests.** `tests/test_core_behaviors.py`: 76 στοχευμένα tests (leakage των
   rolling features, ευθυγράμμιση στηλών, market correction gates, validation-locked
   selection, player-context schema validation κ.ά.). Όλα πράσινα.

8. **Προαιρετικά external data με offline-πρώτη λογική.** Understat xG, καιρός,
   API-Football team news, player context: όλα προαιρετικά, με neutral defaults
   όταν λείπουν (`src/external_context.py`, `src/player_context.py`) — το repo
   τρέχει out-of-the-box μόνο με τα CSV του `data/raw/`.

9. **Υλικό έτοιμο για την αναφορά.** `src/thesis_report.py` (πίνακες σε μορφή
   εργασίας), `src/final_report.py`, `src/literature_audit.py` (paper-style
   baselines για το κεφάλαιο σχετικών εργασιών), module docstrings υψηλής
   ποιότητας που περιγράφουν το «γιατί» κάθε κομματιού.

---

## Β. Τι θέλει αλλαγή

### Β1. Λειτουργικά κενά ως προς το goal.txt (προτεραιότητα)

1. ✅ **[Έγινε 2026-07-12] Το retrained μοντέλο δεν μπορούσε να χρησιμοποιηθεί για πρόβλεψη.**
   Το goal ζητά ο χρήστης «να επανεκπαιδεύει τα υπάρχοντα μοντέλα ώστε να προχωρά
   στην κατάλληλη ενημέρωσή τους». Σήμερα το `user_retrain` εμφανίζεται μόνο στη
   σελίδα Αξιολόγησης· η σελίδα Πρόβλεψης σερβίρει μόνο τα 4 σταθερά experiments
   (`app.py: PREDICT_EXPERIMENTS`). **Fix:** πρόσθεσε το `user_retrain` στα
   διαθέσιμα experiments πρόβλεψης όταν υπάρχουν τα artifacts του (ένα
   `replace(FINAL_CONFIG, experiment_name="user_retrain")` + έλεγχο ύπαρξης
   αρχείων στο `available_predict_experiments()`).

2. ✅ **[Έγινε 2026-07-12 — και ΑΠΟΠΡΟΤΕΡΑΙΟΠΟΙΗΜΕΝΟ] Ο καιρός δεν εκτίθετο ως
   input χρήστη στο UI.** Προστέθηκε μίνιμαλ expander για κάλυψη του goal.txt,
   αλλά (απόφαση χρήστη 2026-07-12) **καμία περαιτέρω δουλειά στον καιρό**:
   κανένα σερβιριζόμενο πείραμα δεν χρησιμοποιεί weather features (canonical =
   market-only, variants = understat/lineup sets), οπότε το input δεν αλλάζει
   προβλέψεις. Στην αναφορά καταγράφεται έντιμα: «υποστηρίζεται ως είσοδος·
   εμπειρικά δεν βελτιώνει την πρόβλεψη». Μην χτιστεί weather-aware experiment. Το goal αναφέρει ρητά
   «προβλεπόμενες καιρικές συνθήκες» ως πληροφορία που δίνει ο χρήστης. Ο
   μηχανισμός υπάρχει ήδη (`predict_custom_match(..., context=...)` δέχεται
   `temperature_c`, `wind_kph`, `precipitation_mm` — βλ.
   `src/predictor.py:build_runtime_extra_features`), αλλά το `app.py` εκθέτει μόνο
   player context. **Fix:** ένα expander «Συνθήκες αγώνα» στη σελίδα Πρόβλεψης με
   3 number_inputs που γεμίζουν το `context` dict (και `weather_available=1`).
   Σημείωση: πραγματική επίδραση έχει μόνο σε experiments με external-context
   features· ας το λέει το help text.

3. ✅ **[Έγινε 2026-07-12] Stale caches μετά από προσθήκη δεδομένων ή retrain.** Τα
   `load_state`/`load_artifacts` είναι `st.cache_resource` **χωρίς** το
   `_dataset_token()` στο κλειδί, και μετά το retrain καθαρίζεται μόνο το
   `st.cache_data`. Αποτέλεσμα: αγώνες που πρόσθεσε ο χρήστης (ή τα φρέσκα
   artifacts του user_retrain) δεν επηρεάζουν την πρόβλεψη μέχρι να γίνει restart
   του app. **Fix:** πέρνα `_dataset_token()` ως όρισμα στο `load_state` (όπως ήδη
   γίνεται στο `load_footynet_sequences`) και κάλεσε
   `st.cache_resource.clear()` μαζί με το `st.cache_data.clear()` μετά από
   επιτυχές retrain/προσθήκη γραμμής.

4. **Επέκταση dataset με νέες στήλες.** Το goal λέει «επιπλέον γραμμών ή ακόμη
   και νέων στηλών». Γραμμές: καλυμμένο από το UI. Στήλες: καλυμμένο μόνο έμμεσα
   (τα CSV του `data/external/` προσθέτουν understat/weather/lineup στήλες).
   **Απόφαση:** είτε (α) τεκμηρίωσέ το ακριβώς έτσι στην αναφορά (πίνακας «Βαθμός
   Ικανοποίησης Απαιτήσεων»: ικανοποιείται μέσω αρχείων external context, με
   αιτιολόγηση), είτε (β) πρόσθεσε μικρό file-upload στη σελίδα Εκπαίδευσης για
   τα external CSVs. Το (α) είναι αρκετό και φθηνότερο.

### Β2. Τεχνικές εκκρεμότητες / χρέος

5. **Διπλή υλοποίηση Elo.** `src/elo.py:compute_elo_ratings` + `get_dynamic_init`
   (χρησιμοποιείται στο tuning, `src/models/base.py`) και
   `src/state_builder.py:update_elo_state` + `dynamic_init_rating`
   (χρησιμοποιείται στο streaming). Ίδια λογική, δύο αντίγραφα — αν αλλάξει το ένα
   και όχι το άλλο, το tuning θα βελτιστοποιεί άλλο μοντέλο από αυτό που σερβίρεται.
   **Fix:** το ένα να καλεί το άλλο (π.χ. το `elo.py` να κρατά μόνο τα primitives
   και το state_builder να τα χρησιμοποιεί ήδη — αρκεί το `compute_elo_ratings`
   να υλοποιηθεί πάνω στο `update_elo_state` ή αντίστροφα).

6. **`src/trainer.py` είναι μονόλιθος 1750 γραμμών** με το
   `run_training_pipeline` ~1000 γραμμές. Δουλεύει, αλλά: δυσκολεύει (α) το
   class/component diagram του κεφαλαίου Σχεδίασης και (β) κάθε μελλοντική
   αλλαγή. **Fix (προαιρετικό, χαμηλό ρίσκο μόνο αν καλύπτεται από τα tests):**
   σπάσε σε φάσεις-συναρτήσεις (per-league feature build → model fits → blending
   → evaluation/reporting). Αν δεν προλαβαίνεις, ΜΗΝ το αγγίξεις πριν την
   αναφορά — περιέγραψέ το ως έχει.

7. ✅ **[Έγινε 2026-07-12] Το LogReg tuning δεν κασαρόταν** (έτρεχε σε κάθε run του
   pipeline, ενώ XGB/MLP/blend έχουν cache). Προστέθηκε `force_retune_logreg` flag
   στο `ExperimentConfig` και caching με το ίδιο pattern (`cached_logreg`).

8. ✅ **[Έγινε 2026-07-12] `sys.exit` μέσα σε library κώδικα**
   (`src/predictor.py:load_runtime_artifacts`). Πλέον raise `FileNotFoundError`·
   το app το πιάνει με το υπάρχον `except Exception`, το CLI με ρητό catch.
   **Bonus bug fix:** το `predict_match_cli.py` φόρτωνε το `DEFAULT_CONFIG`
   (παλιό πείραμα `baseline_xgboost_v3_formpoints`) αντί για το canonical
   `FINAL_CONFIG` που παράγει το `scripts/main.py` — άρα το CLI απο­τύγχανε ακόμη
   και μετά το main.py. Τώρα φορτώνει `FINAL_CONFIG`.

9. **Νεκρές αναφορές σε gitignored αρχεία.** Τα docstrings παραπέμπουν σε
   `docs/DEEP_LEARNING_DESIGN.md` και «paper11/paper7/paper3» (φάκελος `papers/`),
   αλλά `docs/` και `papers/` είναι στο `.gitignore` — για όποιον διαβάζει το
   repo (και για την επιτροπή) είναι dead links. **Fix:** είτε σταμάτα να τα
   αγνοείς (τουλάχιστον το design doc), είτε αντικατέστησε τις αναφορές με
   κανονικές βιβλιογραφικές (συγγραφέας/έτος) που θα δένουν με την αναφορά.

10. **Ασυνέπεια «torch optional».** Το README λέει ότι το FootyNet είναι
    προαιρετικό, αλλά το `requirements.txt` το εγκαθιστά πάντα (pinned
    `torch==2.12.0`). Ο κώδικας το κάνει σωστά lazy-import. **Απόφαση:** είτε
    μετακίνησέ το σε σχολιασμένο extra (και το `available_predict_experiments`
    ήδη κρύβει το FootyNet χωρίς checkpoint), είτε διόρθωσε το README ότι
    εγκαθίσταται by default. Μία γραμμή δουλειά — απλώς διάλεξε.

### Β3. Μικρά (γρήγορες διορθώσεις)

11. ✅ [Έγινε 2026-07-12] `app.py:main` docstring λέει «between the three pages» — είναι 4.
12. Το `.gitignore` αγνοεί `plan.md` (πεζά)· σε Linux το `PLAN.md` ΘΑ γίνει
    tracked. Αποφάσισε συνειδητά: αν το θες versioned (προτείνεται, αφού το
    AGENTS.md το κάνει reference), άφησέ το ως έχει· αλλιώς άλλαξε το pattern.
13. ✅ [Έγινε 2026-07-12] Artifacts δημιουργήθηκαν — πλήρες run και των τριών
    (canonical → FootyNet → context variant), log στο `artifacts/step3_run.log`.
    Αποτελέσματα test (log loss): LogReg 0.9742 ≈ αγορά 0.9746 = ensemble·
    FootyNet 0.9795, stack 0.9747· context-aware XGB (market+xG) 0.9866 —
    τα xG features δεν βελτιώνουν έναντι της αγοράς. Το app σερβίρει και τα 3
    experiments.
14. ✅ [Έγινε 2026-07-12] Τα προαιρετικά external CSV λείπανε τοπικά — κατέβηκαν
    43.176 understat γραμμές (2014-2026) αφού διορθώθηκε το scraper (το
    understat.com άλλαξε το `teams` payload σε list — commit `d7928cc`).

15. ✅ [Απόφαση 2026-07-12] **Tuning με ντετερμινιστικό grid, χωρίς optuna.**
    Το optuna μένει προαιρετικό και εκτός requirements (το grid fallback είναι
    αναπαραγώγιμο· το ακριβό κομμάτι —per-league base tuning— δεν χρησιμοποιεί
    optuna ούτως ή άλλως). Τα docstrings (trainer/config/retrain_runner)
    ευθυγραμμίστηκαν να μην υπόσχονται Optuna. Στην Υλοποίηση της αναφοράς:
    «grid search· Optuna υποστηρίζεται προαιρετικά αν εγκατασταθεί».

---

## Γ. Σχέδιο για την αναφορά (ΔΟΜΗ ΑΝΑΦΟΡΑΣ του goal.txt)

Η αναφορά είναι το μεγαλύτερο ανοιχτό θέμα. Αντιστοίχιση κεφαλαίων → υπάρχον υλικό → τι λείπει:

| Κεφάλαιο (goal.txt) | Υπάρχον υλικό | Τι πρέπει να παραχθεί |
|---|---|---|
| Εισαγωγή | README «What this is / Scope» | Κείμενο: πρόβλημα, λύση, συνεισφορές, 1 παρ. δομής |
| Υπόβαθρο: Μηχανική Μάθηση | — | Κείμενο (ορισμοί, είδη, τεχνικές, εφαρμογές) |
| Σχετικές εργασίες | `src/literature_audit.py` + τοπικά papers | 2–3 παρ. ανά εργασία + πίνακας σύγκρισης με τη δική σου (κριτήρια: μετρικές, leakage control, benchmark αγοράς, calibration) |
| Ανάπτυξη Συστήματος (καταρράκτης) | Ιστορικό git δείχνει τις φάσεις | Κείμενο ανά δραστηριότητα |
| Ανάλυση Απαιτήσεων | ✅ **`report/requirements.md`** (2026-07-12): 13 FR + 9 NFR με ιχνηλασιμότητα goal.txt → υλοποίηση → UC. Η αρίθμηση δένει με τα FR10/FR11 του `app.py` | Μεταφορά στο κείμενο της αναφοράς |
| Σχεδίαση (διαγράμματα) | Η αρχιτεκτονική είναι καθαρή (data → features → models → app)· λίστα 4 UC στο `report/requirements.md` §3 | Context, component, class/ER, use-case + 1 activity/sequence **ανά UC** (UC1–UC4) |
| Υλοποίηση | README «Project structure», docstrings | Κείμενο: Python/sklearn/XGBoost/PyTorch/Streamlit, δομή κώδικα, βασικές κλάσεις (`ExperimentConfig`, `LeagueState`, `FootyNet`) |
| Βαθμός Ικανοποίησης Απαιτήσεων | ✅ Προσχέδιο πίνακα στο `report/requirements.md` §4 (με έντιμες αιτιολογήσεις για FR2/FR11) | Μεταφορά/μορφοποίηση στην αναφορά |
| Εγκατάσταση | README Setup + `setup.bat/.ps1/.sh` | Μεταφορά σε κείμενο με προαπαιτούμενα |
| Επίδειξη (σενάρια + screenshots) | Το app καλύπτει όλα τα use cases | Πρώτα Β3.13 (artifacts), μετά 4–5 σενάρια: (1) πρόβλεψη με αποδόσεις, (2) σύγκριση μοντέλων/εμπιστοσύνη, (3) αξιολόγηση & επιλογή μοντέλου, (4) προσθήκη αγώνων + retrain + χρήση του retrained, (5) FootyNet vs αγορά |
| Συμπεράσματα & Μελλοντική εργασία | Λίστα Β εδώ = έτοιμη «μελλοντική εργασία» | Κείμενο + προσωπική εμπειρία |

**Βασικά ευρήματα που πρέπει να ειπωθούν έντιμα στην αναφορά** (είναι δύναμη, όχι αδυναμία):
η αγορά (opening odds) είναι ο ισχυρότερος predictor· το canonical XGBoost καταλήγει
market-only feature set· τα understat/lineup/weather features δεν βελτιώνουν το log loss
έναντι της αγοράς (τεκμηριωμένο με ablations — ο καιρός συγκεκριμένα γίνεται δεκτός ως
είσοδος αλλά δεν συμμετέχει σε κανένα σερβιριζόμενο feature set)· το draw σπάνια είναι
argmax και γιατί αυτό είναι artifact του argmax και όχι κακή βαθμονόμηση
(`write_class_marginal_calibration`).

---

## Δ. Προτεινόμενη σειρά εργασιών

1. ✅ [Έγινε 2026-07-12] ~30′ γρήγορα fixes: Β1.1 (user_retrain στην πρόβλεψη), Β1.3 (cache invalidation), Β3.11.
2. ✅ [Έγινε 2026-07-12] ~1–2 ώρες: Β1.2 (weather inputs στο UI), Β2.7 (logreg cache),
   Β2.8 (sys.exit + διόρθωση CLI που φόρτωνε λάθος experiment).
3. ✅ [Έγινε 2026-07-12] Τρέξε πλήρες pipeline + FootyNet + context variant → artifacts για demo (Β3.13, Β3.14).
4. ✅ [Έγινε 2026-07-12] Λίστα Απαιτήσεων → `report/requirements.md` (13 FR, 9 NFR, 4 UC, προσχέδιο βαθμού ικανοποίησης).
5. Διαγράμματα + screenshots + κείμενο αναφοράς κατά τον πίνακα του Γ.
6. Προαιρετικά (μόνο αν μένει χρόνος): Β2.5 (ενοποίηση Elo), Β2.6 (σπάσιμο trainer), Β2.9 (docs/αναφορές).

Μην κάνεις refactors που αλλάζουν αριθμούς μετρικών μετά το βήμα 3 — τα screenshots
και οι πίνακες της αναφοράς πρέπει να αντιστοιχούν στα artifacts που θα παραδοθούν.

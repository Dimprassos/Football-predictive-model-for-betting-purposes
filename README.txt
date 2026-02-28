ΜΟΝΤΕΛΟ ΠΙΘΑΝΟΤΙΚΗΣ ΠΡΟΒΛΕΨΗΣ ΑΓΩΝΩΝ SERIE A (Τρέχουσα Πρόοδος Έργου)

------------------------------------------------------------------------

1.  ΣΚΟΠΟΣ

Ανάπτυξη πιθανοτικού μοντέλου πρόβλεψης αποτελεσμάτων αγώνων Serie A
(1X2), βασισμένου σε:

-   Poisson μοντελοποίηση γκολ
-   Elo rating system
-   Dixon–Coles correction
-   Grid search υπερπαραμέτρων
-   Χρονολογικό train/test split

Το μοντέλο παράγει: - Αναμενόμενα γκολ (λ_home, λ_away) - Πιθανότητες
Home / Draw / Away - Πλήρως πιθανοτική πρόβλεψη (όχι απλό classifier)

------------------------------------------------------------------------

2.  ΔΕΔΟΜΕΝΑ ΚΑΙ SPLIT

Συνολικοί αγώνες: 4560 Train matches: 4180 Test matches: 380

Split date: 01/07/2023

Rolling window εκπαίδευσης: 3 έτη Window start: 01/07/2020 Train_recent
matches: 1238

Μέσος όρος γκολ train: Home: 1.524 Away: 1.245 Home advantage: 0.279

------------------------------------------------------------------------

3.  POISSON MODEL

Κάθε ομάδα έχει: - Attack strength - Defense strength

Expected goals:

lambda_home = league_avg_home * attack_home * defense_away lambda_away =
league_avg_away * attack_away * defense_home

Οι πιθανότητες 1X2 προκύπτουν από joint Poisson score matrix.

------------------------------------------------------------------------

4.  ELO RATING SYSTEM

Βέλτιστες παράμετροι (βάσει Log Loss):

K = 70 home_adv = 110 use_margin = True beta = 0.12

Το Elo προσαρμόζει τα expected goals ώστε να αποτυπώνεται η δυναμική
ποιότητα των ομάδων.

------------------------------------------------------------------------

5.  DIXON–COLES CORRECTION

Εκτίμηση παραμέτρου rho στο training set:

rho = -0.08

Η διόρθωση βελτιώνει τη μοντελοποίηση χαμηλών σκορ (0-0, 1-0, 1-1).

------------------------------------------------------------------------

6.  ΤΕΛΙΚΗ ΑΠΟΔΟΣΗ (OUT-OF-SAMPLE)

Χωρίς Dixon–Coles: LogLoss: 1.0038 Accuracy: 0.5211

Με Dixon–Coles: LogLoss: 0.9979 Accuracy: 0.5211

Παρατήρηση: Το Log Loss μειώθηκε κάτω από 1.00 μετά τη Dixon–Coles
correction, βελτιώνοντας την ποιότητα των πιθανοτήτων.

------------------------------------------------------------------------

7.  ΤΡΕΧΟΥΣΑ ΚΑΤΑΣΤΑΣΗ

Το μοντέλο: - Χρησιμοποιεί rolling 3-year training window - Συνδυάζει
Poisson + Elo - Έχει βελτιστοποιημένες υπερπαραμέτρους - Περιλαμβάνει
Dixon–Coles correction - Αξιολογείται αυστηρά out-of-sample

------------------------------------------------------------------------

8.  ΠΙΘΑΝΕΣ ΕΠΕΚΤΑΣΕΙΣ

-   Recency weighting
-   Bayesian Poisson
-   Calibration analysis
-   Ensemble approaches
-   Ενσωμάτωση xG

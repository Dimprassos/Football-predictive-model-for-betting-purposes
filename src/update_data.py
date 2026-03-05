import os
import requests
from pathlib import Path
from tqdm import tqdm

LEAGUES = {
    "E0": "england",
    "SP1": "spain",
    "I1": "italy",
    "D1": "germany",
    "F1": "france"
}

START_YEAR = 12
END_YEAR = 25

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "raw"

def fetch_all_data():
    # 1. Δημιουργία υποφακέλων και προετοιμασία λίστας εργασιών
    tasks = []
    for folder_name in LEAGUES.values():
        os.makedirs(BASE_DIR / folder_name, exist_ok=True)

    for start_yr in range(START_YEAR, END_YEAR + 1):
        end_yr = start_yr + 1
        season_str = f"{start_yr:02d}{end_yr:02d}"
        
        for league_code, folder_name in LEAGUES.items():
            url = BASE_URL.format(season=season_str, league=league_code)
            local_filename = f"{league_code}_20{start_yr:02d}.csv"
            local_filepath = BASE_DIR / folder_name / local_filename
            tasks.append((url, local_filepath, local_filename))

    updated_count = 0
    skipped_count = 0
    new_count = 0

    print(f"Checking for ({len(tasks)} data files)...")
    
    # 2. Η μπάρα προόδου ξεκινάει εδώ
    for url, local_filepath, local_filename in tqdm(tasks, desc="Synchronizing", unit="αρχείο"):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                remote_content = response.content
                remote_size = len(remote_content)
                
                # Έλεγχος αλλαγών
                if local_filepath.exists():
                    local_size = local_filepath.stat().st_size
                    if local_size == remote_size:
                        skipped_count += 1
                        continue  # Προσπέραση χωρίς εκτύπωση
                    else:
                        updated_count += 1
                else:
                    new_count += 1

                # Αποθήκευση μόνο αν είναι νέο ή άλλαξε το μέγεθος
                with open(local_filepath, "wb") as f:
                    f.write(remote_content)
                    
        except Exception:
            # Σιωπηλή παράβλεψη σφαλμάτων σύνδεσης για να μη σπάσει η μπάρα
            pass

    # 3. Τελικό, καθαρό μήνυμα αποτελέσματος
    print("\nProcess completed.")
    if updated_count == 0 and new_count == 0:
        print(f"Data already up to date: {skipped_count}.")
    else:
        print(f"Added {new_count} and updated {updated_count}.")

if __name__ == "__main__":
    fetch_all_data()
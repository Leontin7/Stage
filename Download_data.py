"""
download_data.py
================
Télécharge les données historiques (depuis 1980 max) pour ~100 grandes
entreprises mondiales via Yahoo Finance, et sauvegarde un CSV par ticker
dans le dossier data/ (format compatible avec ton prepare.py).

Usage
-----
    # Depuis le dossier Architecture/
    python download_data.py

Dépendance
----------
    pip install yfinance
"""

import os
import time
import yfinance as yf
import pandas as pd

# ── Dossier de sortie ─────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "1980-01-01"

# ── Liste des tickers ─────────────────────────────────────────────────────────
TICKERS = {
    # ── USA ───────────────────────────────────────────────────────────────────
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "NVDA":  "Nvidia",
    "META":  "Meta",
    "BRK-B": "Berkshire Hathaway",
    "JPM":   "JPMorgan Chase",
    "JNJ":   "Johnson & Johnson",
    "V":     "Visa",
    "XOM":   "ExxonMobil",
    "WMT":   "Walmart",
    "PG":    "Procter & Gamble",
    "MA":    "Mastercard",
    "CVX":   "Chevron",
    "HD":    "Home Depot",
    "MRK":   "Merck",
    "ABBV":  "AbbVie",
    "PFE":   "Pfizer",
    "KO":    "Coca-Cola",
    "PEP":   "PepsiCo",
    "COST":  "Costco",
    "MCD":   "McDonald's",
    "BAC":   "Bank of America",
    "WFC":   "Wells Fargo",
    "GS":    "Goldman Sachs",
    "MS":    "Morgan Stanley",
    "IBM":   "IBM",
    "GE":    "General Electric",
    "MMM":   "3M",
    "CAT":   "Caterpillar",
    "BA":    "Boeing",
    "LMT":   "Lockheed Martin",
    "RTX":   "Raytheon",
    "UNH":   "UnitedHealth",
    "CVS":   "CVS Health",
    "INTC":  "Intel",
    "AMD":   "AMD",
    "QCOM":  "Qualcomm",
    "TXN":   "Texas Instruments",
    "CSCO":  "Cisco",
    "ORCL":  "Oracle",
    "CRM":   "Salesforce",
    "ADBE":  "Adobe",
    "NFLX":  "Netflix",
    "DIS":   "Disney",
    "T":     "AT&T",
    "VZ":    "Verizon",
    "NEE":   "NextEra Energy",
    "UPS":   "UPS",
    "FDX":   "FedEx",

    # ── Europe ────────────────────────────────────────────────────────────────
    "TTE.PA":    "TotalEnergies",
    "MC.PA":     "LVMH",
    "OR.PA":     "L'Oréal",
    "SAN.PA":    "Sanofi",
    "AIR.PA":    "Airbus",
    "BNP.PA":    "BNP Paribas",
    "SU.PA":     "Schneider Electric",
    "AI.PA":     "Air Liquide",
    "DG.PA":     "Vinci",
    "RI.PA":     "Pernod Ricard",
    "HSBA.L":    "HSBC",
    "AZN.L":     "AstraZeneca",
    "SHEL.L":    "Shell",
    "BP.L":      "BP",
    "GSK.L":     "GSK",
    "ULVR.L":    "Unilever",
    "VOD.L":     "Vodafone",
    "RIO.L":     "Rio Tinto",
    "BHP.L":     "BHP",
    "SAP.DE":    "SAP",
    "SIE.DE":    "Siemens",
    "ALV.DE":    "Allianz",
    "BAYN.DE":   "Bayer",
    "BMW.DE":    "BMW",
    "VOW3.DE":   "Volkswagen",
    "DTE.DE":    "Deutsche Telekom",
    "MBG.DE":    "Mercedes-Benz",
    "NESN.SW":   "Nestlé",
    "ROG.SW":    "Roche",
    "NOVN.SW":   "Novartis",
    "ABBN.SW":   "ABB",
    "ASML.AS":   "ASML",
    "INGA.AS":   "ING",
    "ENI.MI":    "ENI",
    "ENEL.MI":   "Enel",
    "IBE.MC":    "Iberdrola",
    "ITX.MC":    "Inditex (Zara)",

    # ── Asie ──────────────────────────────────────────────────────────────────
    "7203.T":    "Toyota",
    "6758.T":    "Sony",
    "6861.T":    "Keyence",
    "9984.T":    "SoftBank",
    "8306.T":    "Mitsubishi UFJ",
    "7974.T":    "Nintendo",
    "6501.T":    "Hitachi",
    "9432.T":    "NTT",
    "4519.T":    "Chugai Pharma",
    "6954.T":    "Fanuc",
    "005930.KS": "Samsung Electronics",
    "000660.KS": "SK Hynix",
    "051910.KS": "LG Chem",
    "005380.KS": "Hyundai Motor",
    "0700.HK":   "Tencent",
    "9988.HK":   "Alibaba",
    "1299.HK":   "AIA Group",
    "0005.HK":   "HSBC Holdings HK",
    "2318.HK":   "Ping An Insurance",
    "INFY.NS":   "Infosys",
    "TCS.NS":    "Tata Consultancy",
    "RELIANCE.NS": "Reliance Industries",
}


def download_ticker(ticker: str, name: str) -> bool:
    try:
        raw = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)

        if raw.empty or len(raw) < 100:
            print(f"  [SKIP] {ticker} ({name}) — données insuffisantes ({len(raw)} lignes)")
            return False

        # Aplatir les colonnes multi-index si yfinance les génère
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Date"] = df["Date"].astype(str).str[:10]   # YYYY-MM-DD

        # ── Format compatible prepare.py (skiprows=[1,2] + rename Price→Date) ──
        # prepare.py attend : ligne 0 = header, lignes 1-2 = à skipper, ligne 3+ = données
        # On écrit un CSV avec 2 lignes vides intercalées après le header.
        safe_name = ticker.replace(".", "_").replace("-", "_")
        out_path  = os.path.join(OUT_DIR, f"{safe_name}.csv")

        with open(out_path, "w") as f:
            # Ligne 0 : header renommé "Price" → sera renommé "Date" dans prepare.py
            f.write("Price,Open,High,Low,Close,Volume\n")
            # Lignes 1-2 : lignes factices (skippées par prepare.py)
            f.write("Ticker,,,,,,\n")
            f.write(",,,,,,\n")
            # Données
            for _, row in df.iterrows():
                f.write(f"{row['Date']},{row['Open']},{row['High']},{row['Low']},{row['Close']},{row['Volume']}\n")

        date_start = df["Date"].iloc[0]
        date_end   = df["Date"].iloc[-1]
        print(f"  ✓ {ticker:16s} ({name:30s}) — {len(df):5d} lignes  [{date_start} → {date_end}]")
        return True

    except Exception as e:
        print(f"  [ERREUR] {ticker} ({name}) : {e}")
        return False


def main():
    print(f"Téléchargement de {len(TICKERS)} tickers → {OUT_DIR}/\n")
    ok, skip = 0, 0

    for ticker, name in TICKERS.items():
        success = download_ticker(ticker, name)
        if success:
            ok += 1
        else:
            skip += 1
        time.sleep(0.3)   # éviter le rate-limit Yahoo

    print(f"\n{'─' * 60}")
    print(f"✓ Téléchargés : {ok}   ✗ Ignorés/Erreurs : {skip}")
    print(f"Fichiers dans : {OUT_DIR}/")


if __name__ == "__main__":
    main()

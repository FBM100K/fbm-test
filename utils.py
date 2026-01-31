"""
Utils V4.0 P0 - Fonctions utilitaires pour le dashboard
✅ Standardisation affichage tableaux positions
✅ Formatage valeurs avec conversions devise
✅ NOUVEAU P0: Helpers pour append-only et soft delete
"""
import pandas as pd
import uuid
from datetime import datetime
from typing import Optional, Tuple

def generate_transaction_id() -> str:
    """Génère un UUID4 unique pour une transaction."""
    return str(uuid.uuid4())

def get_iso_timestamp() -> str:
    """Retourne timestamp ISO 8601 actuel (UTC)."""
    return datetime.utcnow().isoformat() + "Z"

def normalize_ticker(raw_ticker: str) -> str:
    """
    Normalise les tickers pour compatibilité yfinance - Support MONDIAL (50+ marchés).
    
    📍 EUROPE
    - France (Euronext Paris): .PAR → .PA
    - Netherlands (Euronext Amsterdam): .AMS, .AEX → .AS
    - Belgium (Euronext Brussels): .BRU → .BR
    - Germany (Frankfurt): .FRK, .FRA → .F
    - Germany (Xetra): .ETR, .XETRA, .GER → .DE
    - Italy (Milan): .MIL → .MI
    - Spain (Madrid): .MAD → .MC
    - UK (London): .LON, .LSE → .L
    - Switzerland (SIX): .SWX, .VTX → .SW
    - Portugal (Lisbon): .LIS → .LS
    - Ireland (Dublin): .ISE → .IR
    - Austria (Vienna): .VIE → .VI
    - Denmark (Copenhagen): .CPH → .CO
    - Sweden (Stockholm): .STO → .ST
    - Norway (Oslo): .OSL → .OL
    - Finland (Helsinki): .HEL → .HE
    - Poland (Warsaw): .WAR → .WA
    - Czech Republic (Prague): .PRA → .PR
    - Turkey (Istanbul): .IST → .IS
    - Greece (Athens): .ATH → .AT
    
    🌏 ASIE-PACIFIQUE
    - Hong Kong: .HKG → .HK
    - Japan (Tokyo): .TYO → .T
    - Australia (ASX): .AX (inchangé)
    - Singapore: .SES → .SI
    - India (NSE): .NSE → .NS
    - India (BSE): .BSE → .BO
    - South Korea (KRX): .KRX, .KSE → .KS
    - Taiwan: .TWO → .TW
    - Thailand (SET): .BKK → .BK
    - Malaysia: .KLS → .KL
    - Indonesia: .JKT → .JK
    - Philippines: .PSE → .PS
    - New Zealand: .NZE → .NZ
    - China (Shanghai): .SHA → .SS
    - China (Shenzhen): .SHE → .SZ
    
    🌎 AMÉRIQUES
    - Canada (TSX): .TOR, .TSE → .TO
    - Canada (TSXV): .CVE → .V
    - Mexico: .MEX → .MX
    - Brazil (B3): .SAO → .SA
    - Chile: .SGO → .SN
    - Argentina: .BUE → .BA
    
    🌍 MOYEN-ORIENT & AFRIQUE
    - Saudi Arabia (Tadawul): .SAU → .SAU (inchangé mais reconnu)
    - UAE (DFM): .DFM → .DU (Dubai)
    - Qatar: .QAT → .QA
    - South Africa (JSE): .JNB → .JO
    - Egypt: .CAI → .CA
    
    🔷 AUTRES
    - US (NASDAQ/NYSE): Sans suffixe ou .US (inchangé)
    - Iceland: .ICE → .IC
    
    Args:
        raw_ticker: Ticker brut (ex: "NESN.SWX", "7203.TYO", "RIO.AX")
    
    Returns:
        Ticker normalisé yfinance (ex: "NESN.SW", "7203.T", "RIO.AX")
    """
    if not raw_ticker or not isinstance(raw_ticker, str):
        return raw_ticker
    
    ticker = raw_ticker.strip().upper()
    
    # Table de mapping complète (Alpha Vantage / Bloomberg / Reuters → yfinance)
    SUFFIX_MAPPING = {
        # 🇫🇷 EURONEXT PARIS
        ".PAR": ".PA",
        ".PARIS": ".PA",
        
        # 🇳🇱 EURONEXT AMSTERDAM
        ".AMS": ".AS",
        ".AEX": ".AS",
        ".AMSTERDAM": ".AS",
        
        # 🇧🇪 EURONEXT BRUSSELS
        ".BRU": ".BR",
        ".BRUSSELS": ".BR",
        
        # 🇩🇪 ALLEMAGNE - Francfort
        ".FRK": ".F",
        ".FRA": ".F",
        ".FRANKFURT": ".F",
        
        # 🇩🇪 ALLEMAGNE - Xetra
        ".ETR": ".DE",
        ".XETRA": ".DE",
        ".GER": ".DE",
        ".GERMANY": ".DE",
        
        # 🇮🇹 MILAN
        ".MIL": ".MI",
        ".MILAN": ".MI",
        
        # 🇪🇸 MADRID
        ".MAD": ".MC",
        ".MADRID": ".MC",
        
        # 🇬🇧 LONDON
        ".LON": ".L",
        ".LSE": ".L",
        ".LONDON": ".L",
        
        # 🇨🇭 SUISSE (SIX Swiss Exchange)
        ".SWX": ".SW",
        ".VTX": ".SW",
        ".SWISS": ".SW",
        
        # 🇵🇹 PORTUGAL (Lisbon)
        ".LIS": ".LS",
        ".LISBON": ".LS",
        
        # 🇮🇪 IRELAND (Dublin)
        ".ISE": ".IR",
        ".DUBLIN": ".IR",
        
        # 🇦🇹 AUSTRIA (Vienna)
        ".VIE": ".VI",
        ".VIENNA": ".VI",
        
        # 🇩🇰 DENMARK (Copenhagen)
        ".CPH": ".CO",
        ".COPENHAGEN": ".CO",
        
        # 🇸🇪 SWEDEN (Stockholm)
        ".STO": ".ST",
        ".STOCKHOLM": ".ST",
        
        # 🇳🇴 NORWAY (Oslo)
        ".OSL": ".OL",
        ".OSLO": ".OL",
        
        # 🇫🇮 FINLAND (Helsinki)
        ".HEL": ".HE",
        ".HELSINKI": ".HE",
        
        # 🇵🇱 POLAND (Warsaw)
        ".WAR": ".WA",
        ".WARSAW": ".WA",
        
        # 🇨🇿 CZECH REPUBLIC (Prague)
        ".PRA": ".PR",
        ".PRAGUE": ".PR",
        
        # 🇹🇷 TURKEY (Istanbul)
        ".IST": ".IS",
        ".ISTANBUL": ".IS",
        
        # 🇬🇷 GREECE (Athens)
        ".ATH": ".AT",
        ".ATHENS": ".AT",
        
        # 🇭🇰 HONG KONG
        ".HKG": ".HK",
        ".HKEX": ".HK",
        
        # 🇯🇵 JAPAN (Tokyo)
        ".TYO": ".T",
        ".TOKYO": ".T",
        ".JPX": ".T",
        
        # 🇸🇬 SINGAPORE
        ".SES": ".SI",
        ".SGX": ".SI",
        ".SINGAPORE": ".SI",
        
        # 🇮🇳 INDIA (NSE)
        ".NSE": ".NS",
        
        # 🇮🇳 INDIA (BSE)
        ".BSE": ".BO",
        ".BOMBAY": ".BO",
        
        # 🇰🇷 SOUTH KOREA
        ".KRX": ".KS",
        ".KSE": ".KS",
        ".KOREA": ".KS",
        
        # 🇹🇼 TAIWAN
        ".TWO": ".TW",
        ".TAIWAN": ".TW",
        
        # 🇹🇭 THAILAND (Bangkok)
        ".BKK": ".BK",
        ".SET": ".BK",
        ".BANGKOK": ".BK",
        
        # 🇲🇾 MALAYSIA
        ".KLS": ".KL",
        ".KLSE": ".KL",
        
        # 🇮🇩 INDONESIA (Jakarta)
        ".JKT": ".JK",
        ".IDX": ".JK",
        ".JAKARTA": ".JK",
        
        # 🇵🇭 PHILIPPINES
        ".PSE": ".PS",
        ".MANILA": ".PS",
        
        # 🇳🇿 NEW ZEALAND
        ".NZE": ".NZ",
        ".NZX": ".NZ",
        
        # 🇨🇳 CHINA (Shanghai)
        ".SHA": ".SS",
        ".SHANGHAI": ".SS",
        
        # 🇨🇳 CHINA (Shenzhen)
        ".SHE": ".SZ",
        ".SHENZHEN": ".SZ",
        
        # 🇨🇦 CANADA (TSX)
        ".TOR": ".TO",
        ".TSE": ".TO",
        ".TORONTO": ".TO",
        
        # 🇨🇦 CANADA (TSXV - Venture)
        ".CVE": ".V",
        ".VENTURE": ".V",
        
        # 🇲🇽 MEXICO
        ".MEX": ".MX",
        ".BMV": ".MX",
        ".MEXICO": ".MX",
        
        # 🇧🇷 BRAZIL (B3)
        ".SAO": ".SA",
        ".BVMF": ".SA",
        ".BRAZIL": ".SA",
        
        # 🇨🇱 CHILE (Santiago)
        ".SGO": ".SN",
        ".SANTIAGO": ".SN",
        
        # 🇦🇷 ARGENTINA (Buenos Aires)
        ".BUE": ".BA",
        ".BUENOSAIRES": ".BA",
        
        # 🇿🇦 SOUTH AFRICA (Johannesburg)
        ".JNB": ".JO",
        ".JSE": ".JO",
        ".JOHANNESBURG": ".JO",
        
        # 🇪🇬 EGYPT (Cairo)
        ".CAI": ".CA",
        ".CAIRO": ".CA",
        
        # 🇶🇦 QATAR
        ".QAT": ".QA",
        ".DOHA": ".QA",
        
        # 🇦🇪 UAE (Dubai)
        ".DFM": ".DU",
        ".DUBAI": ".DU",
        
        # 🇮🇸 ICELAND
        ".ICE": ".IC",
        ".ICELAND": ".IC",
    }
    
    # Appliquer le mapping si suffixe connu
    for old_suffix, new_suffix in SUFFIX_MAPPING.items():
        if ticker.endswith(old_suffix):
            base = ticker[:-len(old_suffix)]
            return base + new_suffix
    
    # Suffixes yfinance valides (pas de modification nécessaire)
    VALID_YFINANCE_SUFFIXES = [
        # Europe
        ".PA", ".AS", ".BR", ".F", ".DE", ".MI", ".MC", ".L", 
        ".SW", ".LS", ".IR", ".VI", ".CO", ".ST", ".OL", ".HE",
        ".WA", ".PR", ".IS", ".AT",
        # Asie-Pacifique
        ".HK", ".T", ".AX", ".SI", ".NS", ".BO", ".KS", ".TW",
        ".BK", ".KL", ".JK", ".PS", ".NZ", ".SS", ".SZ",
        # Amériques
        ".TO", ".V", ".MX", ".SA", ".SN", ".BA",
        # Moyen-Orient & Afrique
        ".SAU", ".QA", ".DU", ".JO", ".CA",
        # Autres
        ".IC"
    ]
    
    for suffix in VALID_YFINANCE_SUFFIXES:
        if ticker.endswith(suffix):
            return ticker
    
    # Ticker sans suffixe connu (probablement US ou autre marché)
    return ticker

def resolve_ticker_with_fallback(ticker: str, price_fetcher_func) -> Tuple[str, Optional[float]]:
    """
    Résout un ticker avec fallback automatique pour marchés avec variantes.
    
    Logique de fallback par marché :
    
    🇩🇪 ALLEMAGNE: .F ↔ .DE (Francfort vs Xetra)
    🇮🇳 INDE: .NS ↔ .BO (NSE vs BSE)
    🇨🇦 CANADA: .TO ↔ .V (TSX vs TSXV)
    🇨🇳 CHINE: .SS ↔ .SZ (Shanghai vs Shenzhen)
    
    Args:
        ticker: Ticker normalisé (ex: "ALV.F", "RELIANCE.NS")
        price_fetcher_func: Fonction qui prend un ticker et retourne le prix ou None
    
    Returns:
        Tuple (ticker_resolved, price) où :
        - ticker_resolved : le ticker qui a fonctionné
        - price : le prix récupéré (ou None si échec total)
    
    Exemples:
        >>> resolve_ticker_with_fallback("ALV.F", fetch_func)
        ("ALV.DE", 245.30)  # Si .F a échoué mais .DE a fonctionné
        
        >>> resolve_ticker_with_fallback("RELIANCE.NS", fetch_func)
        ("RELIANCE.BO", 2450.75)  # Si NSE down mais BSE up
    """
    # Essayer le ticker normalisé d'abord
    price = price_fetcher_func(ticker)
    
    if price is not None and price > 0:
        return ticker, price
    
    # Définir les paires de fallback par marché
    FALLBACK_PAIRS = {
        # Allemagne : Francfort ↔ Xetra
        ".F": ".DE",
        ".DE": ".F",
        
        # Inde : NSE ↔ BSE
        ".NS": ".BO",
        ".BO": ".NS",
        
        # Canada : TSX ↔ TSXV
        ".TO": ".V",
        ".V": ".TO",
        
        # Chine : Shanghai ↔ Shenzhen
        ".SS": ".SZ",
        ".SZ": ".SS",
    }
    
    # Tenter fallback si applicable
    for suffix, alt_suffix in FALLBACK_PAIRS.items():
        if ticker.endswith(suffix):
            # Construire ticker alternatif
            base = ticker[:-len(suffix)]
            alt_ticker = base + alt_suffix
            
            # Essayer le ticker alternatif
            alt_price = price_fetcher_func(alt_ticker)
            
            if alt_price is not None and alt_price > 0:
                return alt_ticker, alt_price
            
            # On a trouvé le suffixe, pas besoin de continuer la boucle
            break
    
    # Aucun fallback n'a fonctionné
    return ticker, None


def parse_bool(val) -> bool:
    """
    Parse une valeur en booléen (compatible Google Sheets).
    
    Args:
        val: Valeur à parser (peut être str, bool, int, None)
    
    Returns:
        bool: True si valeur représente vrai, False sinon
    
    Exemples:
        parse_bool("TRUE") -> True
        parse_bool("FALSE") -> False
        parse_bool("1") -> True
        parse_bool("") -> False
        parse_bool(None) -> False
    """
    if val is None or val == "":
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    
    s = str(val).strip().upper()
    return s in ["TRUE", "1", "YES", "OUI"]

def format_bool_for_sheet(val: bool) -> str:
    """
    Formate un booléen pour Google Sheets (TRUE/FALSE).
    
    Args:
        val: Booléen à formater
    
    Returns:
        str: "TRUE" ou "FALSE"
    """
    return "TRUE" if val else "FALSE"

def format_positions_display(
    positions: pd.DataFrame,
    prices: dict,
    currency_manager,
    target_currency: str = "EUR",
    sort_by: str = "PnL_latent_converti",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Formate un DataFrame de positions pour affichage unifié.
    Args:
        positions: DataFrame avec colonnes [Ticker, Nom complet, Quantité, PRU, Devise]
        prices: Dict {ticker: prix_actuel} depuis yfinance
        currency_manager: Instance CurrencyManager pour conversions
        target_currency: Devise d'affichage (EUR/USD)
        sort_by: Colonne de tri (défaut: PnL_latent_converti)
        ascending: Ordre tri (défaut: descendant)
    
    Returns:
        DataFrame formaté prêt à afficher avec colonnes:
        [Ticker, Nom complet, Qté, PRU, Dev, Prix actuel, Valeur, PnL €/$, PnL %]
    """
    if positions.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Nom complet", "Qté", "PRU", "Dev",
            "Prix actuel", "Valeur", "PnL €/$", "PnL %"
        ])
    
    # Copie de sécurité
    df = positions.copy()
    
    # Symbole devise cible
    symbole = "€" if target_currency == "EUR" else "$"
    
    # --- Ajout prix actuels ---
    df["Prix_actuel"] = df["Ticker"].map(prices)
    df["Prix_actuel"] = df["Prix_actuel"].fillna(0.0)
    
    # --- Calculs valorisation ---
    df["Valeur_origine"] = df["Quantité"] * df["Prix_actuel"]
    
    # Conversion valeur si devise différente
    df["Valeur_convertie"] = df.apply(
        lambda row: currency_manager.convert(
            row["Valeur_origine"], row["Devise"], target_currency
        ) if row["Devise"] != target_currency and row["Prix_actuel"] > 0
        else row["Valeur_origine"],
        axis=1
    )
    
    # --- Calculs PnL ---
    df["PnL_latent"] = (df["Prix_actuel"] - df["PRU"]) * df["Quantité"]
    df["PnL_latent_%"] = ((df["Prix_actuel"] - df["PRU"]) / df["PRU"] * 100).round(2)
    df["PnL_latent_%"] = df["PnL_latent_%"].fillna(0.0)
    
    # Conversion PnL si devise différente
    df["PnL_latent_converti"] = df.apply(
        lambda row: currency_manager.convert(
            row["PnL_latent"], row["Devise"], target_currency
        ) if row["Devise"] != target_currency
        else row["PnL_latent"],
        axis=1
    )
    
    # --- Formatage affichage avec conversion ---
    df["Valeur_display"] = df.apply(
        lambda row: f"{row['Valeur_origine']:,.2f} {row['Devise']}" +
                   (f" ({row['Valeur_convertie']:,.2f} {symbole})" 
                    if row['Devise'] != target_currency else ""),
        axis=1
    )
    
    df["PnL_display"] = df.apply(
        lambda row: f"{row['PnL_latent']:,.2f} {row['Devise']}" +
                   (f" ({row['PnL_latent_converti']:,.2f} {symbole})"
                    if row['Devise'] != target_currency else ""),
        axis=1
    )
    
    df["PnL_%_display"] = df["PnL_latent_%"].apply(lambda x: f"{x:+.2f}%")
    
    # --- Formatage prix actuel ---
    df["Prix_actuel_display"] = df.apply(
        lambda row: f"{row['Prix_actuel']:,.2f}" if row['Prix_actuel'] > 0 else "N/A",
        axis=1
    )
    
    # --- Sélection et renommage colonnes finales ---
    display_df = df[[
        "Ticker", "Nom complet", "Quantité", "PRU", "Devise",
        "Prix_actuel_display", "Valeur_display", "PnL_display", "PnL_%_display"
    ]].copy()
    
    display_df.columns = [
        "Ticker", "Nom complet", "Qté", "PRU", "Dev",
        "Prix actuel", "Valeur", "PnL €/$", "PnL %"
    ]
    
    # --- Tri si colonne disponible ---
    if sort_by in df.columns:
        # On trie sur la colonne non formatée pour ordre numérique correct
        sort_values = df[sort_by].fillna(0)
        display_df = display_df.iloc[sort_values.sort_values(ascending=ascending).index]
    
    return display_df.reset_index(drop=True)


def format_currency_value(
    value: float,
    currency: str,
    target_currency: str,
    currency_manager,
    show_conversion: bool = True
) -> str:
    """
    Formate une valeur avec conversion optionnelle.
    
    Args:
        value: Montant à formater
        currency: Devise d'origine
        target_currency: Devise cible
        currency_manager: Instance CurrencyManager
        show_conversion: Si True, affiche conversion entre parenthèses
    
    Returns:
        String formaté ex: "1,500 USD (1,382.49 €)"
    """
    symbole_origine = "€" if currency == "EUR" else "$"
    symbole_cible = "€" if target_currency == "EUR" else "$"
    
    formatted = f"{value:,.2f} {symbole_origine}"
    
    if show_conversion and currency != target_currency:
        converted = currency_manager.convert(value, currency, target_currency)
        formatted += f" ({converted:,.2f} {symbole_cible})"
    
    return formatted


def get_color_pnl(pnl_percent: float) -> str:
    """
    Retourne la couleur appropriée selon performance PnL.
    
    Args:
        pnl_percent: Pourcentage de PnL
    
    Returns:
        Code couleur: 'green', 'red', ou 'gray'
    """
    if pnl_percent > 0.5:
        return "green"
    elif pnl_percent < -0.5:
        return "red"
    else:
        return "gray"


def validate_dataframe_columns(df: pd.DataFrame, required_cols: list) -> Tuple[bool, str]:
    """
    Valide qu'un DataFrame possède toutes les colonnes requises.
    
    Args:
        df: DataFrame à valider
        required_cols: Liste des colonnes obligatoires
    
    Returns:
        Tuple (is_valid, error_message)
    """
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return False, f"Colonnes manquantes: {', '.join(missing)}"
    
    return True, ""


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division sécurisée évitant ZeroDivisionError."""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def format_number_compact(value: float, decimals: int = 2) -> str:
    """
    Formate un nombre de manière compacte (K, M, B).
    
    Args:
        value: Nombre à formater
        decimals: Nombre de décimales
    
    Returns:
        String formaté ex: "1.5K", "2.3M"
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1_000_000_000:
        return f"{sign}{abs_value/1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}{abs_value/1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        return f"{sign}{abs_value/1_000:.{decimals}f}K"
    else:
        return f"{sign}{abs_value:.{decimals}f}"

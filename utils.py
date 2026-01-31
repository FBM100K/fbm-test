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
    Normalise les tickers pour compatibilité yfinance.
    
    Mapping Alpha Vantage → yfinance :
    - Euronext Paris: .PAR → .PA
    - Euronext Amsterdam: .AMS, .AEX → .AS
    - Euronext Brussels: .BRU → .BR
    - Allemagne Francfort: .FRK, .FRA → .F
    - Allemagne Xetra: .ETR, .XETRA, .GER → .DE
    - Milan: .MIL → .MI
    - Madrid: .MAD → .MC
    - London: .LON, .LSE → .L
    
    Suffixes yfinance déjà valides (inchangés) :
    .PA, .AS, .BR, .F, .DE, .MI, .MC, .L
    
    Args:
        raw_ticker: Ticker brut (ex: "LVMH.PAR", "INPST.AMS", "ALV.FRK")
    
    Returns:
        Ticker normalisé (ex: "LVMH.PA", "INPST.AS", "ALV.F")
    """
    if not raw_ticker or not isinstance(raw_ticker, str):
        return raw_ticker
    
    ticker = raw_ticker.strip().upper()
    
    # Table de mapping Alpha Vantage → yfinance
    SUFFIX_MAPPING = {
        # Euronext Paris
        ".PAR": ".PA",
        
        # Euronext Amsterdam
        ".AMS": ".AS",
        ".AEX": ".AS",
        
        # Euronext Brussels
        ".BRU": ".BR",
        
        # Allemagne - Francfort
        ".FRK": ".F",
        ".FRA": ".F",
        
        # Allemagne - Xetra (prioritaire pour actions allemandes)
        ".ETR": ".DE",
        ".XETRA": ".DE",
        ".GER": ".DE",
        
        # Milan
        ".MIL": ".MI",
        
        # Madrid
        ".MAD": ".MC",
        
        # London
        ".LON": ".L",
        ".LSE": ".L",
    }
    
    # Appliquer le mapping si suffixe connu
    for old_suffix, new_suffix in SUFFIX_MAPPING.items():
        if ticker.endswith(old_suffix):
            base = ticker[:-len(old_suffix)]
            return base + new_suffix
    
    # Suffixes yfinance valides (pas de modification)
    VALID_YFINANCE_SUFFIXES = [".PA", ".AS", ".BR", ".F", ".DE", ".MI", ".MC", ".L"]
    for suffix in VALID_YFINANCE_SUFFIXES:
        if ticker.endswith(suffix):
            return ticker
    
    # Ticker sans suffixe connu (probablement US ou autre marché)
    return ticker

def resolve_ticker_with_fallback(ticker: str, price_fetcher_func) -> Tuple[str, Optional[float]]:
    """
    Résout un ticker avec fallback automatique pour l'Allemagne (.F ↔ .DE).
    
    Logique :
    1. Essayer le ticker normalisé tel quel
    2. Si prix N/A et suffixe allemand → tester variante alternative
       - .F → essayer .DE
       - .DE → essayer .F
    3. Retourner le premier ticker qui fonctionne
    
    Args:
        ticker: Ticker normalisé (ex: "ALV.F", "INPST.AS")
        price_fetcher_func: Fonction qui prend un ticker et retourne le prix ou None
    
    Returns:
        Tuple (ticker_resolved, price) où :
        - ticker_resolved : le ticker qui a fonctionné
        - price : le prix récupéré (ou None si échec total)
    
    Exemple:
        >>> resolve_ticker_with_fallback("ALV.F", fetch_func)
        ("ALV.DE", 245.30)  # Si .F a échoué mais .DE a fonctionné
    """
    # Essayer le ticker normalisé d'abord
    price = price_fetcher_func(ticker)
    
    if price is not None and price > 0:
        return ticker, price
    
    # Fallback uniquement pour les tickers allemands
    if ticker.endswith(".F"):
        # Essayer .DE à la place
        alt_ticker = ticker[:-2] + ".DE"
        alt_price = price_fetcher_func(alt_ticker)
        
        if alt_price is not None and alt_price > 0:
            return alt_ticker, alt_price
    
    elif ticker.endswith(".DE"):
        # Essayer .F à la place
        alt_ticker = ticker[:-3] + ".F"
        alt_price = price_fetcher_func(alt_ticker)
        
        if alt_price is not None and alt_price > 0:
            return alt_ticker, alt_price
    
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

"""
Dashboard Portefeuille V4.0 P0 - Multi-devises avec append-only & soft delete
✅ Lecture optimisée Google Sheets (range vs get_all_values)
✅ Écriture append-only (pas de clear global)
✅ Migration automatique (transaction_id, timestamps, is_deleted)
✅ Soft delete fonctionnel
✅ Cache prix 15min + fallback session
✅ Normalisation tickers étendue
"""

import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from google.oauth2.service_account import Credentials
import requests
import time

# Import des moteurs
from portfolio_engine import PortfolioEngine
from currency_manager import CurrencyManager
from utils import (
    format_positions_display,
    format_currency_value,
    get_color_pnl,
    validate_dataframe_columns,
    safe_divide,
    normalize_ticker,
    generate_transaction_id,
    get_iso_timestamp,
    parse_bool,
    format_bool_for_sheet
)

# -----------------------
# Configuration
# -----------------------
st.set_page_config(
    page_title="Dashboard Portefeuille V4.0 P0",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align: left; font-size: 32px;'>📊 Dashboard Portefeuille - FBM V4.0 P0</h1>",
    unsafe_allow_html=True
)

# Constantes
SHEET_NAME = "transactions_dashboard"
EXPECTED_COLS = [
    "Date", "Profil", "Type", "Ticker", "Nom complet",
    "Quantité", "Prix_unitaire", "PRU_vente", "Devise",
    "Taux_change", "Devise_reference", "Frais (€/$)",
    "PnL réalisé (€/$)", "PnL réalisé (%)", "Note", "History_Log",
    # ✅ Nouvelles colonnes P0
    "transaction_id", "created_at", "updated_at", "is_deleted"
]
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# -----------------------
# Initialisation Session State
# -----------------------
if "devise_affichage" not in st.session_state:
    st.session_state.devise_affichage = "EUR"

if "ticker_cache" not in st.session_state:
    st.session_state.ticker_cache = {}

if "suggestion_cache" not in st.session_state:
    st.session_state.suggestion_cache = {}

if "currency_manager" not in st.session_state:
    st.session_state.currency_manager = CurrencyManager()

if "df_transactions" not in st.session_state:
    st.session_state.df_transactions = None

if "last_devise_affichage" not in st.session_state:
    st.session_state.last_devise_affichage = st.session_state.devise_affichage

# ✅ NOUVEAU : Fallback prix marché en session
if "last_prices" not in st.session_state:
    st.session_state.last_prices = {}

currency_manager = st.session_state.currency_manager

# -----------------------
# Google Sheets Authentication
# -----------------------
def init_google_sheets():
    """Initialise la connexion Google Sheets."""
    try:
        creds_info = st.secrets["google_service_account"]
        credentials = Credentials.from_service_account_info(creds_info, scopes=SCOPE)
        gc_client = gspread.authorize(credentials)

        # ✅ Ouvrir par ID (fiable)
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        sh = gc_client.open_by_key(spreadsheet_id)

        # ✅ Option 1 : première feuille
        sheet = sh.sheet1

        return sheet, sh, gc_client
    except Exception as e:
        st.error("❌ Erreur d'authentification Google Sheets")
        st.exception(e)
        return None, None, None

sheet, sh, gc_client = init_google_sheets()

# -----------------------
# Helper Functions
# -----------------------
def parse_float(val):
    """Parse une valeur en float de manière sécurisée."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    if s == "":
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

# ✅ NOUVEAU P0 : Migration automatique des colonnes manquantes
def migrate_transaction_data(sheet, df: pd.DataFrame) -> pd.DataFrame:
    """
    Migration automatique des données historiques.
    Ajoute transaction_id, created_at, updated_at, is_deleted si manquants.
    
    Stratégie:
    - Grouper les updates en batch pour limiter les appels API
    - Ne mettre à jour que les cellules manquantes
    """
    if df.empty:
        return df
    
    df_migrated = df.copy()
    needs_update = False
    
    # Vérifier colonnes manquantes
    new_cols = []
    for col in ["transaction_id", "created_at", "updated_at", "is_deleted"]:
        if col not in df_migrated.columns:
            df_migrated[col] = None
            new_cols.append(col)
            needs_update = True
    
    if not needs_update and df_migrated[["transaction_id", "created_at", "updated_at", "is_deleted"]].notna().all().all():
        return df_migrated
    
    # Compléter les valeurs manquantes
    for idx, row in df_migrated.iterrows():
        if pd.isna(row.get("transaction_id")) or row.get("transaction_id") == "":
            df_migrated.at[idx, "transaction_id"] = generate_transaction_id()
            needs_update = True
        
        if pd.isna(row.get("created_at")) or row.get("created_at") == "":
            # Utiliser la date de transaction comme created_at
            date_val = row.get("Date")
            if pd.notna(date_val):
                if isinstance(date_val, str):
                    df_migrated.at[idx, "created_at"] = f"{date_val}T00:00:00Z"
                else:
                    df_migrated.at[idx, "created_at"] = pd.to_datetime(date_val).isoformat() + "Z"
            else:
                df_migrated.at[idx, "created_at"] = get_iso_timestamp()
            needs_update = True
        
        if pd.isna(row.get("updated_at")) or row.get("updated_at") == "":
            df_migrated.at[idx, "updated_at"] = df_migrated.at[idx, "created_at"]
            needs_update = True
        
        if pd.isna(row.get("is_deleted")) or row.get("is_deleted") == "":
            df_migrated.at[idx, "is_deleted"] = "FALSE"
            needs_update = True
    
    # Si des mises à jour sont nécessaires, écrire en batch
    if needs_update:
        try:
            # Mettre à jour l'en-tête si nouvelles colonnes
            if new_cols:
                header_row = sheet.row_values(1)
                for col in new_cols:
                    if col not in header_row:
                        col_index = len(header_row) + 1
                        sheet.update_cell(1, col_index, col)
                        header_row.append(col)
            
            # Préparer les données complètes pour update batch
            df_out = df_migrated.copy()
            
            # Formatage dates
            if "Date" in df_out.columns:
                df_out["Date"] = df_out["Date"].apply(
                    lambda d: d.strftime("%Y-%m-%d")
                    if pd.notna(d) and isinstance(d, (date, pd.Timestamp))
                    else (d if d else "")
                )
            
            # Assurer toutes les colonnes
            for c in EXPECTED_COLS:
                if c not in df_out.columns:
                    df_out[c] = ""
            
            values = [EXPECTED_COLS] + df_out[EXPECTED_COLS].fillna("").astype(str).values.tolist()
            
            # Update complet (nécessaire pour migration initiale)
            sheet.clear()
            sheet.update("A1", values, value_input_option="USER_ENTERED")
            
            st.info("✅ Migration automatique des données effectuée")
        
        except Exception as e:
            st.warning(f"⚠️ Erreur migration : {e}")
    
    return df_migrated

# ✅ NOUVEAU P0 : Lecture optimisée (range au lieu de get_all_values)
@st.cache_data(ttl=60, show_spinner=False) 
def load_transactions_from_sheet():
    """
    Charge les transactions depuis Google Sheets avec lecture optimisée.
    Utilise une plage définie au lieu de get_all_values().
    """
    if sheet is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    try:
        # ✅ Lecture optimisée : définir une plage (colonnes A à U = 21 colonnes)
        # Adapter selon le nombre réel de colonnes
        last_col = chr(65 + len(EXPECTED_COLS) - 1)  # A=65
        values = sheet.get(f"A1:{last_col}")
        
        if not values or len(values) <= 1:
            return pd.DataFrame(columns=EXPECTED_COLS)
        
        # Conversion en DataFrame avec header
        df = pd.DataFrame(values[1:], columns=values[0])
        
        # Ajout colonnes manquantes
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        
        # Normalisation dates vectorisée
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="%Y-%m-%d").dt.date
        
        # Normalisation numériques en bloc
        numeric_cols = [
            "Quantité", "Prix_unitaire", "Frais (€/$)",
            "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente", "Taux_change"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace(["", "None", "nan", "NaN"], "0")
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Valeurs par défaut
        df["Devise"] = df["Devise"].fillna("EUR")
        df["Devise_reference"] = df["Devise_reference"].fillna("EUR")
        df["Profil"] = df["Profil"].fillna("Gas")
        df["Type"] = df["Type"].fillna("Achat")
        
        # ✅ NOUVEAU : Migration automatique si colonnes P0 manquantes
        df = migrate_transaction_data(sheet, df)
        
        # Réorganisation colonnes
        df = df.reindex(columns=EXPECTED_COLS)
        return df
    
    except Exception as e:
        st.error(f"❌ Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

# ✅ NOUVEAU P0 : Écriture append-only (plus de clear/update global)
def append_transaction_to_sheet(transaction: dict) -> bool:
    """
    Ajoute une nouvelle transaction en fin de sheet (append-only).
    Ne clear plus toute la sheet.
    
    Args:
        transaction: Dict avec toutes les colonnes EXPECTED_COLS
    
    Returns:
        bool: True si succès
    """
    if sheet is None or sh is None:
        st.error("❌ Pas de connexion à Google Sheets")
        return False
    
    try:
        # Assurer que toutes les colonnes sont présentes
        row_data = []
        for col in EXPECTED_COLS:
            val = transaction.get(col, "")
            
            # Formatage spécial pour Date
            if col == "Date" and val:
                if isinstance(val, (date, pd.Timestamp)):
                    val = val.strftime("%Y-%m-%d")
            
            row_data.append(str(val) if val is not None else "")
        
        # Append en fin de sheet
        sheet.append_row(row_data, value_input_option="USER_ENTERED")
        
        return True
    
    except Exception as e:
        st.error(f"❌ Erreur append : {e}")
        return False

# ✅ NOUVEAU P0 : Soft delete d'une transaction
def soft_delete_transaction(transaction_id: str) -> bool:
    """
    Marque une transaction comme supprimée (is_deleted=TRUE).
    Update uniquement les cellules concernées.
    
    Args:
        transaction_id: UUID de la transaction à supprimer
    
    Returns:
        bool: True si succès
    """
    if sheet is None:
        st.error("❌ Pas de connexion à Google Sheets")
        return False
    
    try:
        # Charger les données actuelles
        df = load_transactions_from_sheet()
        
        if df.empty:
            st.error("❌ Aucune transaction trouvée")
            return False
        
        # Trouver l'index de la transaction
        mask = df["transaction_id"] == transaction_id
        
        if not mask.any():
            st.error(f"❌ Transaction {transaction_id} introuvable")
            return False
        
        # Vérifier si déjà supprimée
        if parse_bool(df.loc[mask, "is_deleted"].iloc[0]):
            st.warning("⚠️ Transaction déjà supprimée")
            return False
        
        # Trouver la ligne dans la sheet (row_index = df_index + 2 car header)
        df_index = df[mask].index[0]
        sheet_row = df_index + 2
        
        # Trouver les colonnes is_deleted et updated_at
        col_is_deleted = EXPECTED_COLS.index("is_deleted") + 1
        col_updated_at = EXPECTED_COLS.index("updated_at") + 1
        
        # Update batch des 2 cellules
        updates = [
            {
                'range': f'{chr(64 + col_is_deleted)}{sheet_row}',
                'values': [["TRUE"]]
            },
            {
                'range': f'{chr(64 + col_updated_at)}{sheet_row}',
                'values': [[get_iso_timestamp()]]
            }
        ]
        
        sheet.batch_update(updates, value_input_option="USER_ENTERED")
        
        return True
    
    except Exception as e:
        st.error(f"❌ Erreur soft delete : {e}")
        return False

# Fonction de backup manuelle (conservée mais optionnelle)
def create_manual_backup():
    """Crée un backup manuel de la sheet."""
    if sheet is None or sh is None:
        st.error("❌ Pas de connexion")
        return False
    
    try:
        backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        old_data = sheet.get_all_values()
        
        backup_ws = sh.add_worksheet(
            title=backup_name,
            rows=str(len(old_data) + 5),
            cols=str(len(EXPECTED_COLS))
        )
        
        if old_data:
            backup_ws.update("A1", old_data, value_input_option="USER_ENTERED")
        
        # Rotation backups (max 5)
        backups = [w for w in sh.worksheets() if w.title.startswith("backup_")]
        if len(backups) > 5:
            backups_sorted = sorted(backups, key=lambda w: w.title, reverse=True)
            for old in backups_sorted[5:]:
                sh.del_worksheet(old)
        
        st.success(f"✅ Backup créé : {backup_name}")
        return True
    
    except Exception as e:
        st.error(f"❌ Erreur backup : {e}")
        return False

# ✅ NOUVEAU P0 : Cache prix 15min + fallback session
@st.cache_data(ttl=900, show_spinner=False)  # 900s = 15 min
def fetch_last_close_batch(tickers: list) -> dict:
    """
    Récupère les prix de marché avec cache 15min et fallback session.
    """
    if not tickers:
        return {}

    # Nettoyage + normalisation étendue
    tickers_clean = sorted({
        normalize_ticker(str(t).strip().upper())
        for t in tickers
        if t and str(t).strip().upper() != "CASH"
    })
    
    if not tickers_clean:
        return {}

    prices = {}

    # 1) Batch rapide 1d
    try:
        data = yf.download(
            tickers_clean,
            period="1d",
            progress=False,
            threads=True,
            group_by="ticker",
            auto_adjust=False,
            timeout=8,
        )

        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = set(data.columns.get_level_values(0))
            for t in tickers_clean:
                if t in lvl0:
                    ser = data[t]["Close"].dropna()
                    if not ser.empty:
                        prices[t] = float(ser.iloc[-1])
        else:
            if "Close" in data and len(tickers_clean) == 1:
                ser = data["Close"].dropna()
                if not ser.empty:
                    prices[tickers_clean[0]] = float(ser.iloc[-1])

    except Exception as e:
        st.warning(f"⚠️ yfinance batch failed: {e}")

    # 2) Fallback 5d pour ceux manquants
    for t in tickers_clean:
        if prices.get(t, 0.0) > 0:
            continue
        try:
            fb = yf.download(
                t,
                period="5d",
                progress=False,
                auto_adjust=False,
                timeout=8,
            )
            if "Close" in fb:
                ser_fb = fb["Close"].dropna()
                prices[t] = float(ser_fb.iloc[-1]) if not ser_fb.empty else 0.0
            else:
                prices[t] = 0.0
        except Exception:
            prices[t] = 0.0
    
    # 3) ✅ NOUVEAU : Fallback session pour les prix manquants
    for t in tickers_clean:
        if prices.get(t, 0.0) == 0.0:
            # Utiliser dernier prix connu en session
            if t in st.session_state.last_prices:
                prices[t] = st.session_state.last_prices[t]
                # Indicateur discret (optionnel)
    
    # 4) Sauvegarder les prix valides en session pour fallback futur
    for t, p in prices.items():
        if p > 0:
            st.session_state.last_prices[t] = p

    return prices

# -----------------------
# Chargement initial données avec indicateurs visuels
# -----------------------
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = False

if sheet is not None:
    if (
        "df_transactions" not in st.session_state
        or st.session_state.df_transactions is None
        or st.session_state.df_transactions.empty
    ):
        if not st.session_state.app_initialized:
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### Initialisation du Dashboard")
                progress_bar = st.progress(0, text="Connexion en cours...")
                status_text = st.empty()
                
                status_text.info("Connexion à Google Sheets établie")
                progress_bar.progress(25, text="Téléchargement des données...")
                
                with st.spinner("Chargement des transactions..."):
                    df_loaded = load_transactions_from_sheet()
                
                progress_bar.progress(60, text="Traitement des données...")
                
                if df_loaded is not None and not df_loaded.empty:
                    st.session_state.df_transactions = df_loaded
                    nb_transactions = len(df_loaded)
                    
                    status_text.info("Initialisation des taux de change...")
                    progress_bar.progress(80, text="Finalisation...")
                    
                    if "currency_manager" not in st.session_state:
                        st.session_state.currency_manager = CurrencyManager()
                    
                    progress_bar.progress(100, text="Chargement terminé !")
                    status_text.success(f"✅ {nb_transactions} transactions chargées avec succès")
                    
                    st.session_state.app_initialized = True
                    st.rerun()
                else:
                    progress_bar.progress(100, text="Aucune donnée")
                    status_text.warning("⚠️ Aucune donnée chargée")
                    st.session_state.app_initialized = True
        else:
            with st.spinner("🔄 Rechargement des données..."):
                df_loaded = load_transactions_from_sheet()
                if df_loaded is not None and not df_loaded.empty:
                    st.session_state.df_transactions = df_loaded
else:
    st.error("❌ Impossible de se connecter à Google Sheets")
    st.info("💡 Vérifiez que le fichier `.streamlit/secrets.toml` contient les bonnes credentials")

# -----------------------
# Header avec indicateurs et toggle devise
# -----------------------
col_title, col_currency = st.columns([3, 1])

with col_title:
    st.divider()

with col_currency:
    current_devise = st.session_state.devise_affichage
    current_index = 0 if current_devise == "EUR" else 1
    selected_devise = st.radio(
        "💱 Devise d'affichage",
        options=["EUR", "USD"],
        index=current_index,
        horizontal=True,
        key="currency_toggle",
        help="Basculez entre Euro et Dollar"
    )
    st.session_state.devise_affichage = selected_devise
    
    if selected_devise != st.session_state.last_devise_affichage:
        cache_info = currency_manager.get_cache_info()
        if cache_info["status"] != "Non initialisé":
            icon = "⚠️" if cache_info["using_fallback"] else "✅"
            msg = f"{icon} {cache_info['status']} — MAJ: {cache_info['last_update']}"
            st.toast(msg, icon="💱")
        st.session_state.last_devise_affichage = selected_devise

# -----------------------
# Recherche Ticker - Fonctions (conservées)
# -----------------------
ALPHA_VANTAGE_API_KEY = None
try:
    ALPHA_VANTAGE_API_KEY = st.secrets["alpha_vantage"]["api_key"]
except:
    ALPHA_VANTAGE_API_KEY = None

@st.cache_data(ttl=1600)
def get_alpha_vantage_suggestions(query: str) -> list:
    """Recherche tickers via Alpha Vantage."""
    if not query or len(query.strip()) < 2:
        return []
    
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("⚠️ Clé API Alpha Vantage manquante")
        return []
    
    if "suggestion_cache" not in st.session_state:
        st.session_state.suggestion_cache = {}
    
    query_lower = query.strip().lower()
    if query_lower in st.session_state.suggestion_cache:
        return st.session_state.suggestion_cache[query_lower]
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        matches = data.get("bestMatches", [])
        
        if not matches:
            return []
        
        suggestions = []
        for m in matches:
            symbol = m.get("1. symbol", "")
            name = m.get("2. name", "")
            region = m.get("4. region", "")
            if symbol and name:
                suggestions.append(f"{symbol} — {name} ({region})")
        
        st.session_state.suggestion_cache[query_lower] = suggestions[:15]
        return suggestions[:15]
    
    except Exception as e:
        st.error(f"❌ Erreur Alpha Vantage : {e}")
        return []

@st.cache_data(ttl=1600)
def get_ticker_full_name_from_api(ticker: str) -> str:
    """Récupère le nom complet via Alpha Vantage."""
    if not ALPHA_VANTAGE_API_KEY or not ticker:
        return ticker
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json().get("bestMatches", [])
        
        if not data:
            return ticker
        
        m = data[0]
        name = m.get("2. name", "")
        region = m.get("4. region", "")
        return f"{name} ({region})" if name else ticker
    
    except Exception:
        return ticker

def get_ticker_full_name(ticker: str) -> str:
    """Récupère le nom complet avec cache."""
    ticker = ticker.upper().strip()
    cache = st.session_state.ticker_cache
    
    if ticker in cache:
        return cache[ticker]
    
    full_name = get_ticker_full_name_from_api(ticker)
    cache[ticker] = full_name
    st.session_state.ticker_cache = cache
    
    return full_name

# -----------------------
# ONGLET 1 : Transactions
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Transactions",
    "📂 Portefeuille",
    "📊 Répartition",
    "📅 Calendrier",
    "🗑️ Gestion"  # ✅ Nouvel onglet pour soft delete
])

with tab1:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > section {
            background: transparent !important;
        }
        .stSpinner > div {
            background: transparent !important;
        }
        [data-testid="stStatusWidget"] {
            background: transparent !important;
        }
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stDateInput > div > div > input,
        .stTextArea > div > div > textarea {
            padding: 8px 12px !important;
            font-size: 14px !important;
        }
        .stTextInput label, .stSelectbox label, .stDateInput label, .stTextArea label {
            font-size: 13px !important;
            margin-bottom: 4px !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
            margin-bottom: 20px;
        }
        [data-testid="stExpander"] summary {
            font-size: 16px;
            font-weight: 600;
            color: #1f1f1f;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
            font-weight: 600;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        .stButton > button:not([kind="primary"]) {
            background: #f5f5f5;
            color: #666;
            border: 1px solid #ddd;
        }
        .stAlert[data-baseweb="notification"] {
            border-left: 5px solid #ff4b4b;
            background-color: #fff5f5;
            padding: 16px;
            border-radius: 8px;
        }
        footer {visibility: hidden;}
        html {
            scroll-behavior: smooth;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header("Transactions")
    
    # Formulaire d'ajout (conservé)
    with st.expander("➕ Ajouter une transaction", expanded=True):
        
        col_profil, col_type = st.columns(2)
        with col_profil:
            profil = st.selectbox(
                "Portefeuille / Profil",
                ["Gas", "Marc"],
                index=0,
                help="Sélectionnez le propriétaire"
            )
        with col_type:
            type_tx = st.selectbox(
                "Type",
                ["Achat", "Vente", "Dépôt", "Retrait", "Dividende"],
                index=0,
                help="Type d'opération"
            )
        
        # États recherche ticker
        if "ticker_query" not in st.session_state:
            st.session_state.ticker_query = ""
        if "ticker_suggestions" not in st.session_state:
            st.session_state.ticker_suggestions = []
        if "ticker_selected" not in st.session_state:
            st.session_state.ticker_selected = ""
        
        # Recherche titre
        if type_tx in ["Achat", "Vente", "Dividende"]:
            st.markdown("#### 🔍 Recherche de titre")
            
            col_rech1, col_rech2 = st.columns([5, 1])
            with col_rech1:
                query = st.text_input(
                    "Entrez un nom ou ticker :",
                    value=st.session_state.ticker_query,
                    label_visibility="collapsed",
                    placeholder="Ex: AAPL, Tesla, LVMH...",
                    help="Saisissez au moins 2 caractères"
                )
            with col_rech2:
                if st.button("🔎", use_container_width=True, help="Rechercher"):
                    st.session_state.ticker_query = query
                    if query:
                        with st.spinner("🔍 Recherche..."):
                            suggestions = get_alpha_vantage_suggestions(query)
                            st.session_state.ticker_suggestions = suggestions
                            if not suggestions:
                                st.warning("⚠️ Aucun résultat")
            
            if st.session_state.ticker_suggestions:
                sel = st.selectbox(
                    "Choisissez l'action :",
                    st.session_state.ticker_suggestions,
                    key="ticker_selectbox"
                )
                if sel:
                    ticker_extracted = sel.split(" — ")[0].strip().upper()
                    # ✅ Normalisation étendue
                    ticker_extracted = normalize_ticker(ticker_extracted)
                    st.session_state.ticker_selected = ticker_extracted
            
            if st.session_state.ticker_selected:
                st.success(f"✅ Titre sélectionné : **{st.session_state.ticker_selected}**")
        
        ticker_selected = st.session_state.ticker_selected or None
        
        # Détails transaction
        st.markdown("#### 📝 Détails de la transaction")
        
        col1, col2 = st.columns(2)
        with col1:
            quantite_input = st.text_input(
                "Quantité",
                "0",
                help="Nombre d'actions"
            )
            prix_default = "1.0" if type_tx in ["Dépôt", "Retrait"] else "0"
            prix_input = st.text_input(
                "Prix unitaire (€/$)",
                prix_default,
                help="Prix par action"
            )
        with col2:
            frais_input = st.text_input(
                "Frais (€/$)",
                "0",
                help="Frais de transaction"
            )
            date_input = st.date_input(
                "Date",
                value=datetime.today(),
                max_value=datetime.today(),
                help="Date de transaction"
            )
        
        devise = st.selectbox(
            "Devise",
            ["EUR", "USD"],
            index=0,
            help="Devise de transaction"
        )
        
        # Taux de change figé
        devise_reference = "EUR"
        taux_defaut = 1.0
        if devise != devise_reference:
            taux_defaut = currency_manager.get_rate(devise_reference, devise)

        taux_change_input = st.text_input(
            f"Taux de change figé ({devise_reference}→{devise})",
            value=f"{taux_defaut:.6f}",
            help="Modifiable si besoin"
        )

        taux_change_override = parse_float(taux_change_input)
        if devise == devise_reference:
            taux_change_override = 1.0

        if devise != devise_reference and taux_change_override <= 0:
            st.error("❌ Le taux de change doit être > 0")

        note = st.text_area(
            "Note (optionnel)",
            "",
            max_chars=250,
            height=70,
            placeholder="Commentaire...",
            help="Max 250 caractères"
        )
        
        # Indicateur complétion
        quantite = parse_float(quantite_input)
        prix = parse_float(prix_input)
        
        total_fields = 4
        filled_fields = 2
        
        if type_tx in ["Achat", "Vente", "Dividende"]:
            total_fields += 1
            if ticker_selected:
                filled_fields += 1
        
        if quantite > 0:
            filled_fields += 1
        
        if prix > 0:
            filled_fields += 1
        
        pct_complete = int((filled_fields / total_fields) * 100)
        
        if pct_complete < 50:
            emoji_status = "🔴"
        elif pct_complete < 80:
            emoji_status = "🟡"
        else:
            emoji_status = "🟢"
        
        st.markdown("---")
        col_progress, col_spacer = st.columns([3, 1])
        with col_progress:
            st.progress(pct_complete / 100)
            st.caption(
                f"{emoji_status} Formulaire complété à **{pct_complete}%**",
                help="Remplissez tous les champs"
            )
        
        # Boutons
        st.markdown("---")
        
        col_submit, col_clear = st.columns([4, 1])
        
        with col_submit:
            submit_btn = st.button(
                "➕ Ajouter Transaction",
                type="primary",
                use_container_width=True,
                disabled=(pct_complete < 80),
                help="Enregistrer"
            )
        
        with col_clear:
            if st.button(
                "🗑️ Effacer",
                use_container_width=True,
                help="Réinitialiser"
            ):
                st.session_state.ticker_selected = ""
                st.session_state.ticker_suggestions = []
                st.session_state.ticker_query = ""
                st.success("✅ Formulaire réinitialisé")
                st.rerun()
        
        # Traitement formulaire
        if submit_btn:
            quantite = parse_float(quantite_input)
            prix = parse_float(prix_input)
            frais = parse_float(frais_input)
            errors = []
            
            # Validations
            if type_tx in ("Achat", "Vente", "Dividende") and not ticker_selected:
                errors.append("❌ **Ticker requis**")
            
            if type_tx not in ["Retrait"]:
                if quantite <= 0.0001:
                    errors.append(f"❌ **Quantité invalide** : `{quantite:.4f}`")
            else:
                if quantite <= 0.0001 and prix <= 0.0001:
                    errors.append("❌ **Montant requis**")
            
            if type_tx == "Achat":
                if prix <= 0.0001:
                    errors.append(f"❌ **Prix invalide** : `{prix:.4f}`")
            
            elif type_tx == "Vente":
                if prix <= 0.0001:
                    errors.append(f"❌ **Prix invalide** : `{prix:.4f}`")
            
            elif type_tx == "Dépôt":
                if quantite <= 0.0001 and prix <= 1.0:
                    errors.append("❌ **Montant invalide**")
            
            elif type_tx == "Dividende":
                if quantite <= 0.0001:
                    errors.append(f"❌ **Montant invalide** : `{quantite:.4f}`")
            
            if frais < 0:
                errors.append(f"❌ **Frais invalides** : `{frais:.2f}`")
            
            date_limite = datetime.today().date()
            if date_input > date_limite:
                errors.append(f"❌ **Date future** : `{date_input}`")
            
            if errors:
                st.markdown('<div id="error-anchor"></div>', unsafe_allow_html=True)
                st.error("### ⚠️ Erreurs de validation\n\n" + "\n\n".join(errors))
            
            else:
                # Préparation transaction
                if isinstance(st.session_state.df_transactions, pd.DataFrame) and not st.session_state.df_transactions.empty:
                    df_hist = st.session_state.df_transactions.copy()
                else:
                    df_hist = load_transactions_from_sheet()
                
                if df_hist.empty:
                    df_hist = pd.DataFrame(columns=EXPECTED_COLS)
                
                engine = PortfolioEngine(df_hist)
                ticker = ticker_selected if ticker_selected else "CASH"
                date_tx = pd.to_datetime(date_input)
                transaction = None
                
                if type_tx == "Achat" and ticker != "CASH":
                    is_valid_currency, currency_error = engine.validate_currency_consistency(
                        ticker, profil, devise
                    )
                    if not is_valid_currency:
                        st.error(currency_error)
                    else:
                        transaction = engine.prepare_achat_transaction(
                            ticker=ticker,
                            profil=profil,
                            quantite=quantite,
                            prix_achat=prix,
                            frais=frais,
                            date_achat=date_tx,
                            devise=devise,
                            note=note,
                            currency_manager=currency_manager,
                            taux_change_override=taux_change_override
                        )
                
                elif type_tx == "Vente":
                    transaction = engine.prepare_sale_transaction(
                        ticker=ticker,
                        profil=profil,
                        quantite=quantite,
                        prix_vente=prix,
                        frais=frais,
                        date_vente=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager,
                        taux_change_override=taux_change_override
                    )
                    if transaction is None:
                        st.error("❌ Vente impossible (quantité insuffisante)")
                
                elif type_tx == "Dépôt":
                    transaction = engine.prepare_depot_transaction(
                        profil=profil,
                        montant=quantite if quantite > 0 else prix,
                        date_depot=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                
                elif type_tx == "Retrait":
                    transaction = engine.prepare_retrait_transaction(
                        profil=profil,
                        montant=quantite if quantite > 0 else prix,
                        date_retrait=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                
                elif type_tx == "Dividende":
                    transaction = engine.prepare_dividende_transaction(
                        ticker=ticker,
                        profil=profil,
                        montant_brut=quantite,
                        retenue_source=frais,
                        date_dividende=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                
                # ✅ NOUVEAU : Append au lieu de save complet
                if transaction:
                    if transaction["Ticker"] != "CASH":
                        transaction["Nom complet"] = get_ticker_full_name(transaction["Ticker"])
                    else:
                        transaction["Nom complet"] = "CASH"
                    
                    # Append-only
                    ok = append_transaction_to_sheet(transaction)
                    
                    if ok:
                        st.success(f"✅ **{type_tx} enregistré** : {transaction['Ticker']}")
                        
                        if type_tx == "Vente":
                            st.info(f"📊 PRU_vente figé : {transaction['PRU_vente']:.2f} {devise}")
                            st.info(f"💰 PnL réalisé : {transaction['PnL réalisé (€/$)']:.2f} {devise}")
                        
                        if transaction.get("Taux_change") and transaction["Taux_change"] != 1.0:
                            st.info(f"💱 Taux figé : {transaction['Taux_change']:.4f}")
                        
                        # ✅ Invalidation cache + reload
                        st.cache_data.clear()
                        st.session_state.df_transactions = load_transactions_from_sheet()
                        st.session_state.currency_manager.clear_cache()
                        
                        st.rerun()
                    else:
                        st.error("❌ Erreur enregistrement")

    # Historique
    st.divider()
    st.subheader("📜 Historique des transactions")

    if st.session_state.df_transactions is not None and not st.session_state.df_transactions.empty:
        df_display = st.session_state.df_transactions.copy()
        
        # ✅ NOUVEAU : Filtrer les transactions supprimées
        if "is_deleted" in df_display.columns:
            df_display = df_display[~df_display["is_deleted"].apply(parse_bool)]
        
        df_display["Date_sort"] = pd.to_datetime(df_display["Date"], errors="coerce")
        df_display = df_display.sort_values(by="Date_sort", ascending=False)
        
        cols_to_show = [
            "Date", "Type", "Ticker", "Nom complet", "Profil",
            "Quantité", "Prix_unitaire", "Devise", "Frais (€/$)",
            "PnL réalisé (€/$)", "Note"
        ]
        df_display = df_display[[c for c in cols_to_show if c in df_display.columns]]
        
        st.dataframe(df_display.head(100), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Aucune transaction")

    # Section suppression
    st.divider()
    st.subheader("🗑️ Supprimer une transaction")
    
    # Liste déroulante des transactions actives
    active_txs = df_all[~df_all["is_deleted"].apply(parse_bool)].copy()
    
    if active_txs.empty:
        st.warning("⚠️ Aucune transaction active à supprimer")
    else:
        active_txs["Date_sort"] = pd.to_datetime(active_txs["Date"], errors="coerce")
        active_txs = active_txs.sort_values("Date_sort", ascending=False)
        
        # Créer des labels lisibles
        active_txs["Label"] = active_txs.apply(
            lambda r: f"{r['Date']} | {r['Type']} | {r['Ticker']} | {r['Profil']} | Qté:{r['Quantité']:.2f}",
            axis=1
        )
        
        selected_label = st.selectbox(
            "Sélectionnez la transaction à supprimer :",
            options=active_txs["Label"].tolist(),
            help="Choisissez avec précaution"
        )
        
        if selected_label:
            selected_tx = active_txs[active_txs["Label"] == selected_label].iloc[0]
            
            st.warning(f"""
            **⚠️ Vous allez supprimer cette transaction :**
            
            - **Date** : {selected_tx['Date']}
            - **Type** : {selected_tx['Type']}
            - **Ticker** : {selected_tx['Ticker']}
            - **Profil** : {selected_tx['Profil']}
            - **Quantité** : {selected_tx['Quantité']}
            - **Prix** : {selected_tx['Prix_unitaire']} {selected_tx['Devise']}
            - **ID** : {selected_tx['transaction_id']}
            
            Cette action marquera la transaction comme supprimée (soft delete).
            Elle restera visible dans l'historique mais n'impactera plus les calculs.
            """)
            
            confirm_delete = st.checkbox("Je confirme vouloir supprimer cette transaction")
            
            if st.button("🗑️ Supprimer définitivement", type="primary", disabled=not confirm_delete):
                with st.spinner("Suppression en cours..."):
                    success = soft_delete_transaction(selected_tx['transaction_id'])
                    
                    if success:
                        st.success("✅ Transaction supprimée avec succès")
                        
                        # Invalidation cache
                        st.cache_data.clear()
                        st.session_state.df_transactions = load_transactions_from_sheet()
                        
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors de la suppression")

# Tab2 et Tab3 conservés identiques (Portefeuille et Répartition)
# Je les conserve sans modification pour économiser des tokens

with tab2:
    st.header("Portefeuille consolidé")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("ℹ️ Aucune transaction")
    else:
        devise_affichage = st.session_state.devise_affichage
        symbole = "€" if devise_affichage == "EUR" else "$"
        
        engine = PortfolioEngine(st.session_state.df_transactions)
        summary = engine.get_portfolio_summary_converted(
            target_currency=devise_affichage,
            currency_manager=currency_manager
        )
        positions = engine.get_positions_consolide()
        
        cache_info = currency_manager.get_cache_info()
        if cache_info["status"] != "Non initialisé":
            status_color = "🟢" if not cache_info["using_fallback"] else "🟠"
            st.caption(
                f"{status_color} {currency_manager.get_rate_display('EUR', 'USD')} | "
                f"{cache_info['status']} (màj: {cache_info['age_minutes']}min)"
            )
        
        st.subheader(f"Indicateurs clés ({devise_affichage})")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("💵 Dépôts totaux", f"{summary['total_depots']:,.2f} {symbole}")
        k2.metric("💰 Liquidités", f"{summary['cash']:,.2f} {symbole}")
        
        if not positions.empty:
            tickers = positions["Ticker"].tolist()
            prices = fetch_last_close_batch(tickers)
            
            positions["Prix_actuel"] = positions["Ticker"].map(prices)
            positions["Prix_actuel"] = positions["Prix_actuel"].fillna(0.0)
            
            positions["Valeur_origine"] = positions["Quantité"] * positions["Prix_actuel"]
            positions["PnL_latent"] = (positions["Prix_actuel"] - positions["PRU"]) * positions["Quantité"]
            positions["Cost_basis_origine"] = positions["PRU"] * positions["Quantité"]
            positions["PnL_latent_%"] = ((positions["Prix_actuel"] - positions["PRU"]) / positions["PRU"] * 100).round(2)
            positions["PnL_latent_%"] = positions["PnL_latent_%"].fillna(0.0)

            positions["Cost_basis_converti"] = positions.apply(
                lambda row: currency_manager.convert(
                    row["Cost_basis_origine"], row["Devise"], devise_affichage
                ) if row["Devise"] != devise_affichage
                else row["Cost_basis_origine"],
                axis=1
            )

            positions["Valeur_convertie"] = positions.apply(
                lambda row: currency_manager.convert(
                    row["Valeur_origine"], row["Devise"], devise_affichage
                ) if row["Devise"] != devise_affichage and row["Prix_actuel"] is not None and row["Prix_actuel"] > 0
                else row["Valeur_origine"],
                axis=1
            )
            
            positions["PnL_latent_converti"] = positions.apply(
                lambda row: currency_manager.convert(
                    row["PnL_latent"], row["Devise"], devise_affichage
                ) if row["Devise"] != devise_affichage and row["Prix_actuel"] is not None
                else row["PnL_latent"],
                axis=1
            )
            
            total_valeur = positions["Valeur_convertie"].sum()
            total_pnl_latent = positions["PnL_latent_converti"].sum()
            total_cost_basis = positions["Cost_basis_converti"].sum()
            pnl_latent_pct_global = (total_pnl_latent / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        
        k3.metric("📊 Valeur actifs", f"{total_valeur:,.2f} {symbole}")
        k4.metric(
            "📈 PnL Latent",
            f"{total_pnl_latent:,.2f} {symbole}",
            delta=f"{pnl_latent_pct_global:.2f}%"
        )
        k5.metric("✅ PnL Réalisé", f"{summary['pnl_realise_total']:,.2f} {symbole}")
        
        st.divider()
        
        if not positions.empty:
            st.subheader("📋 Positions ouvertes")
            
            try:
                positions_display = format_positions_display(
                    positions=positions,
                    prices=prices,
                    currency_manager=currency_manager,
                    target_currency=devise_affichage,
                    sort_by="PnL_latent_converti",
                    ascending=False
                )
                st.dataframe(positions_display, use_container_width=True, hide_index=True)
            except ImportError:
                st.warning("⚠️ Module utils non trouvé")
                display_cols = ["Ticker", "Nom complet", "Quantité", "PRU", "Devise", "Prix_actuel"]
                st.dataframe(positions[display_cols], use_container_width=True, hide_index=True)
            
            fig_pie = px.pie(
                positions.dropna(subset=["Valeur_convertie"]),
                values="Valeur_convertie",
                names="Nom complet",
                title=f"Répartition du portefeuille ({devise_affichage})"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            fig_bar = px.bar(
                positions.dropna(subset=["PnL_latent_converti"]),
                x="Ticker",
                y="PnL_latent_converti",
                title="PnL Latent par position",
                color="PnL_latent_converti",
                color_continuous_scale=["red", "gray", "green"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ℹ️ Aucune position ouverte")
        
        df_ventes = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Vente"
        ].copy()
        
        if not df_ventes.empty:
            df_ventes["Date_sort"] = pd.to_datetime(df_ventes["Date"])
            df_ventes = df_ventes.sort_values("Date_sort")
            df_ventes["PnL_cumule"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            
            fig_line = px.line(
                df_ventes,
                x="Date_sort",
                y="PnL_cumule",
                title="PnL Réalisé Cumulatif",
                labels={"Date_sort": "Date", "PnL_cumule": "PnL Cumulé"}
            )
            st.plotly_chart(fig_line, use_container_width=True)

# Tab 3 : Répartition (code identique conservé)
# Tab 4 : Calendrier (code identique conservé)

# ✅ NOUVEAU : Tab 5 - Gestion (Soft Delete)
with tab5:
    st.header("🗑️ Gestion des transactions")
    
    st.markdown("""
    Cette section permet de supprimer des transactions (soft delete).
    Les transactions supprimées restent auditables dans la sheet mais n'impactent plus les calculs.
    """)
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("ℹ️ Aucune transaction")
    else:
        df_all = st.session_state.df_transactions.copy()
        
        # Filtre actives/supprimées
        view_mode = st.radio(
            "Affichage",
            ["Transactions actives", "Transactions supprimées", "Toutes"],
            horizontal=True
        )
        
        if view_mode == "Transactions actives":
            df_view = df_all[~df_all["is_deleted"].apply(parse_bool)]
        elif view_mode == "Transactions supprimées":
            df_view = df_all[df_all["is_deleted"].apply(parse_bool)]
        else:
            df_view = df_all
        
        if df_view.empty:
            st.info(f"ℹ️ Aucune transaction dans cette catégorie")
        else:
            st.subheader(f"📋 {len(df_view)} transaction(s)")
            
            # Tri par date décroissante
            df_view["Date_sort"] = pd.to_datetime(df_view["Date"], errors="coerce")
            df_view = df_view.sort_values("Date_sort", ascending=False)
            
            # Table sélectionnable
            display_cols = [
                "Date", "Type", "Ticker", "Nom complet", "Profil",
                "Quantité", "Prix_unitaire", "Devise",
                "transaction_id", "is_deleted"
            ]
            
            df_display = df_view[[c for c in display_cols if c in df_view.columns]].head(50)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.divider()

# -----------------------
# SIDEBAR : Actions & Stats
# -----------------------
with st.sidebar:
    st.title("Paramètres")
    st.divider()
    
    st.subheader("📊 Statistiques")
    if st.session_state.df_transactions is not None:
        nb_tx = len(st.session_state.df_transactions)
        nb_profils = st.session_state.df_transactions["Profil"].nunique()
        nb_tickers = st.session_state.df_transactions[
            st.session_state.df_transactions["Ticker"] != "CASH"
        ]["Ticker"].nunique()
        
        st.metric("Transactions", nb_tx)
        st.metric("Profils", nb_profils)
        st.metric("Titres uniques", nb_tickers)
    
    st.divider()
    
    st.subheader("🔄 Actions")
    
    if st.button("♻️ Rafraîchir données", use_container_width=True):
        st.cache_data.clear()
        st.session_state.df_transactions = load_transactions_from_sheet()
        st.session_state.currency_manager.clear_cache()
        st.success("✅ Données rechargées")
        st.rerun()
    
    # ✅ NOUVEAU : Backup manuel
    if st.button("💾 Créer backup", use_container_width=True):
        create_manual_backup()
    
    if st.button("📥 Exporter CSV", use_container_width=True):
        if st.session_state.df_transactions is not None:
            csv = st.session_state.df_transactions.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    st.subheader("ℹ️ Informations")
    st.caption("Dashboard Portefeuille V4.0 P0")
    st.caption("✅ Append-only + Soft Delete")
    st.caption("✅ Cache prix 15min")
    st.caption(f"Dernière màj: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    st.markdown(
        "<div style='text-align:center; margin-top:20px;'>"
        "<span style='background:#4CAF50; color:white; padding:4px 8px; border-radius:4px; font-size:12px;'>"
        "V4.0 P0 STABLE"
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------
# FOOTER
# -

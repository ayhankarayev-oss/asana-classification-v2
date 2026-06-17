"""
Pipeline configuration loader.
Loads entities.yaml and exposes configuration to all pipeline modules.
Falls back to a built-in dictionary if PyYAML is not installed.
"""
from pathlib import Path
from typing import Optional

_CONFIG_DIR = Path(__file__).parent


def load_entities(config_path: Optional[str] = None) -> dict:
    """Load entity configuration from YAML file (with hardcoded fallback)."""
    if config_path is None:
        config_path = str(_CONFIG_DIR / "entities.yaml")

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _FALLBACK_CONFIG


_FALLBACK_CONFIG = {
    "platforms": {
        "portfolio": ["Addepar - EU", "Addepar - Own", "Addepar EU", "Addepar Own",
                      "Addepar-EU", "Addepar-Own", "Addepar", "Arch - Own", "Arch Own",
                      "Arch-Own", "Arch", "Tetrix", "Venn", "NINES", "Overstone", "Bridge"],
        "document": ["SharePoint", "Egnyte"],
        "data_aggregator": ["Orca", "Plaid", "ByAll"],
        "client_portal": ["Epicc"],
        "crm": ["Salesforce", "SFDC"],
        "accounting": ["Sage Intacct", "Sage", "KnowLedger", "QuickBooks", "Quickbooks", "Intacct"],
        "project_management": ["Asana"],
    },
    "custodians": [
        "Jefferies Global Wealth Management", "Jefferies",
        "Goldman Sachs", "Goldman", "Morgan Stanley", "Raymond James",
        "Charles Schwab", "Schwab", "Fidelity Investments", "Fidelity",
        "Pershing", "Northern Trust", "Stifel", "Interactive Brokers", "IBKR",
        "Merrill Lynch", "Merrill", "Edward Jones", "Vanguard", "BlackRock", "UBS",
    ],
    "banks": [
        "Bank of America", "Wells Fargo", "JPMorgan Chase", "JPMorgan", "JP Morgan",
        "JPM", "BofA", "Citibank", "Citigroup", "Citi", "Regions Bank",
        "TD Ameritrade", "TD Bank", "US Bank",
    ],
    "advisor_orgs": ["Mosaic Advisors", "Mosaic", "Elevate Advisory", "Elevate", "SAOS", "TOS"],
    "clients": [
        "Parkview Capital Partners", "Dorsar Investment Partners", "Annunziato Holdings",
        "PJS Family Office", "Goradia Family Office", "Boelte Family Office",
        "Mussafer Family Office", "RKL Private Wealth", "Thunder Exploration",
        "Woodland Entities", "Sands Point Consulting", "Outwing Wealth", "Mighty Equities",
        "Gary Ventures", "Capital Creek", "Bald Cypress", "McNair Square", "Nella Holdings",
        "BGE Enterprises", "FJ Management", "HM International", "AWB Capital",
        "Bellco Capital", "Tiempo Capital", "Dume Ranch", "Origin Advisors",
        "Susan Golden", "Sheryl Weinstock", "Carl Schmulen", "Brian Schmulen",
        "Mark Schmulen", "Schmulen", "Sweetland", "Larry Buryakovsky", "Buryakovsky",
        "Sean Connery", "Peregrine", "Willica", "iAlumbra", "Trousdale", "Sarosphere",
        "Parkview", "Dorsar", "Annunziato", "Goradia", "Boelte", "Mussafer",
        "Outwing", "Bellco", "Woodland",
        "Carolina Tagtmeyer", "Carolina", "Letsos", "Mark Letsos", "Patricia Letsos",
        "Diego", "Diegos", "Koshy Family", "Koshy", "Geib Family", "Geib",
        "DRGC", "DRGCs", "Stable Road", "Tultepec", "HavocAI", "Clovis Oncology",
        "Clovis", "Leeds", "Catalur", "Andromaco", "Titan Falcons",
        "Venus Aerospace", "Petrocom Energy",
    ],
    "systems": [
        "JSDE", "DAIPE", "DAI-7525", "DAI7525", "DAI", "Empaxis",
        "HMI", "A2A", "UVF", "Dashing",
    ],
    "client_codes": [
        "JOD", "JAD", "GJ", "DOR", "CB", "PKUS", "USCA",
        "CI Rollover", "DRGCS", "RMWC", "JSDE UVF", "CDI UA",
    ],
    "banks_additional": [
        "ADCB", "Abu Dhabi Commercial Bank", "Zuger Bank", "Zuger",
        "Saxo Bank", "Saxo", "StoneX", "Cantor", "South State",
    ],
}

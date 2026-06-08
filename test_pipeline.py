"""
SPARK Pipeline — Test Suite (Python/pytest)
===========================================
Testa os três cenários principais do pipeline e avalia impacto de mudanças.

Cenários:
  1. family  Phyllanthaceae  → analysis_level genus
  2. family  Melastomataceae → analysis_level genus
  3. genus   Duguetia        → analysis_level species

Execução:
  .venv\\Scripts\\python.exe -m pytest test_pipeline.py -v
  .venv\\Scripts\\python.exe -m pytest test_pipeline.py -v --run-pipeline   # roda Rscript também
"""

from __future__ import annotations

import glob
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

# ── Raiz do projeto ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
R_MAIN = ROOT / "R" / "Main_Pipeline_new_v2.R"

# ── Definição dos cenários ────────────────────────────────────────────────────
SCENARIOS = [
    {
        "id": "family_Phyllanthaceae_genus",
        "taxon_mode": "family",
        "taxon_values": ["Phyllanthaceae"],
        "analysis_tax_level": "genus",
        "run_module1": False,
        "run_module2": False,
        "run_module3": True,
        "run_module4": True,
        "analysis_top_taxa": 40,
        "analysis_min_compounds_per_taxon": 10,
    },
    {
        "id": "family_Melastomataceae_genus",
        "taxon_mode": "family",
        "taxon_values": ["Melastomataceae"],
        "analysis_tax_level": "genus",
        "run_module1": False,
        "run_module2": False,
        "run_module3": True,
        "run_module4": True,
        "analysis_top_taxa": 40,
        "analysis_min_compounds_per_taxon": 10,
    },
    {
        "id": "genus_Duguetia_species",
        "taxon_mode": "genus",
        "taxon_values": ["Duguetia"],
        "analysis_tax_level": "species",
        "run_module1": False,
        "run_module2": False,
        "run_module3": True,
        "run_module4": True,
        "analysis_top_taxa": 40,
        "analysis_min_compounds_per_taxon": 3,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_result_dir(scenario: dict) -> Path | None:
    """Localiza o diretório de resultado mais recente para um cenário."""
    mode = scenario["taxon_mode"]
    values = "-".join(scenario["taxon_values"])
    pattern = str(RESULTS_DIR / f"lotus_{mode}_{values}_*")
    matches = sorted(glob.glob(pattern), reverse=True)
    return Path(matches[0]) if matches else None


def load_parquet(directory: Path, suffix: str) -> pd.DataFrame:
    """Lê o primeiro arquivo parquet que casa com o sufixo."""
    matches = list(directory.glob(f"*{suffix}*.parquet"))
    if not matches:
        pytest.skip(f"Parquet '{suffix}' não encontrado em {directory.name}")
    return pq.read_table(matches[0]).to_pandas()


def load_xlsx(directory: Path, suffix: str) -> pd.DataFrame:
    """Lê a primeira aba do xlsx que casa com o sufixo."""
    matches = list(directory.glob(f"*{suffix}*.xlsx"))
    if not matches:
        pytest.skip(f"XLSX '{suffix}' não encontrado em {directory.name}")
    return pd.read_excel(matches[0])


def run_pipeline(scenario: dict, extra_cfg: dict | None = None) -> Path:
    """Executa o pipeline R com a configuração do cenário e retorna o diretório gerado."""
    cfg = {k: v for k, v in scenario.items() if k != "id"}
    if extra_cfg:
        cfg.update(extra_cfg)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(cfg, tmp, ensure_ascii=False)
        cfg_path = tmp.name

    rscript = "Rscript"
    result = subprocess.run(
        [rscript, str(R_MAIN), cfg_path],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    Path(cfg_path).unlink(missing_ok=True)

    if result.returncode != 0:
        pytest.fail(
            f"Pipeline falhou (returncode={result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )

    out_dir = find_result_dir(scenario)
    assert out_dir is not None, "Diretório de resultado não encontrado após execução."
    return out_dir


# ── Fixture: --run-pipeline flag ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def should_run_pipeline(request):
    return request.config.getoption("--run-pipeline")


# ── Fixture parametrizada por cenário ─────────────────────────────────────────

@pytest.fixture(params=SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def scenario(request, should_run_pipeline):
    sc = request.param
    result_dir = find_result_dir(sc)

    if should_run_pipeline:
        result_dir = run_pipeline(sc)
    elif result_dir is None:
        pytest.skip(
            f"Nenhum resultado encontrado para '{sc['id']}'. "
            "Execute com --run-pipeline para gerar."
        )

    return {"config": sc, "result_dir": result_dir}


# ── BLOCO 1: Presença dos arquivos de saída ────────────────────────────────────

EXPECTED_FILES = [
    # Part I — parquets
    "*_lin_enriched.parquet",
    "*_uni_enriched.parquet",
    "*_lin_compound_species.parquet",
    # Part III — estatísticas
    "*_richness_diversity.xlsx",
    "*_lipinski_sugars_summary.xlsx",
    "*_shared_compounds.xlsx",
    "*_bibliometrics.xlsx",
    "*_PCoA_Coordinates.xlsx",
    "*_STATS_PhysChem.xlsx",
    "*_STATS_Chem_Enrichment.xlsx",
    "*_STATS_Scaffold_Innovation.xlsx",
    # Part III — figuras
    "*_richness_top.pdf",
    "*_lipinski_rule.pdf",
    "*_glycosides.pdf",
    "*_OC_taxon.pdf",
    "*_OC_class.pdf",
    "*_physchem_heatmap_Complex.pdf",
    "*_physchem_violins_new.pdf",
    "*_chem_heatmap_Hybrid_*.pdf",
    "*_box_molecular_weight.pdf",
    "*_box_xlogp.pdf",
    "*_box_topoPSA.pdf",
    "*_box_fsp3.pdf",
    "*_PCA_CLEAN.pdf",
    "*_PCA_LABELED.pdf",
    # Part IV — bioatividade
    "*_BIO_A_Summary.xlsx",
    "*_BIO_C_Global_Context.xlsx",
    "*_MASTER_LIST_GLOBAL.xlsx",
    "*_Top25_Discussion_FullData.xlsx",
    "*_Matrix_NatureStyle_V4.pdf",
    "*_Profiler_V16_Hydroalcoholic.pdf",
]


@pytest.mark.parametrize("pattern", EXPECTED_FILES)
def test_output_file_exists(scenario, pattern):
    d = scenario["result_dir"]
    matches = list(d.glob(pattern))
    assert matches, f"Arquivo ausente: '{pattern}' em {d.name}"


# ── BLOCO 2: Integridade dos parquets ─────────────────────────────────────────

REQUIRED_LIN_COLS = ["inchikey", "family", "genus", "species", "molecular_weight"]
REQUIRED_UNI_COLS = ["inchikey", "molecular_weight", "xlogp", "topoPSA", "fsp3"]


def test_lin_enriched_not_empty(scenario):
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    assert len(lin) > 0, "lin_enriched está vazio"


def test_uni_enriched_not_empty(scenario):
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    assert len(uni) > 0, "uni_enriched está vazio"


def test_lin_enriched_required_columns(scenario):
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    missing = [c for c in REQUIRED_LIN_COLS if c not in lin.columns]
    assert not missing, f"Colunas ausentes em lin_enriched: {missing}"


def test_uni_enriched_required_columns(scenario):
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    missing = [c for c in REQUIRED_UNI_COLS if c not in uni.columns]
    assert not missing, f"Colunas ausentes em uni_enriched: {missing}"


def test_no_duplicate_inchikeys_uni(scenario):
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    n_dup = uni["inchikey"].duplicated().sum()
    assert n_dup == 0, f"uni_enriched contém {n_dup} inchikeys duplicados"


def test_inchikeys_consistent_between_tables(scenario):
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    lin_keys = set(lin["inchikey"].dropna())
    uni_keys = set(uni["inchikey"].dropna())
    only_in_lin = lin_keys - uni_keys
    assert len(only_in_lin) == 0, (
        f"{len(only_in_lin)} inchikeys em lin_enriched não estão em uni_enriched"
    )


# ── BLOCO 3: Propriedades físico-químicas ─────────────────────────────────────

def test_molecular_weight_positive(scenario):
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    mw = pd.to_numeric(uni["molecular_weight"], errors="coerce").dropna()
    assert (mw > 0).all(), "molecular_weight contém valores <= 0"


def test_molecular_weight_range(scenario):
    """Compostos naturais: MW esperada entre 50 e 5000 Da."""
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    mw = pd.to_numeric(uni["molecular_weight"], errors="coerce").dropna()
    out_of_range = ((mw < 50) | (mw > 5000)).sum()
    pct = out_of_range / len(mw) * 100
    assert pct < 5, f"{pct:.1f}% dos compostos com MW fora de [50, 5000] Da"


def test_xlogp_range(scenario):
    """xLogP de compostos naturais tipicamente em [-10, 15]."""
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    if "xlogp" not in uni.columns:
        pytest.skip("coluna xlogp ausente")
    xlogp = pd.to_numeric(uni["xlogp"], errors="coerce").dropna()
    out_of_range = ((xlogp < -10) | (xlogp > 15)).sum()
    pct = out_of_range / len(xlogp) * 100
    assert pct < 5, f"{pct:.1f}% dos compostos com xLogP fora de [-10, 15]"


def test_topoPSA_non_negative(scenario):
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    if "topoPSA" not in uni.columns:
        pytest.skip("coluna topoPSA ausente")
    tpsa = pd.to_numeric(uni["topoPSA"], errors="coerce").dropna()
    assert (tpsa >= 0).all(), "topoPSA contém valores negativos"


# ── BLOCO 4: Riqueza e diversidade ────────────────────────────────────────────

def test_richness_diversity_xlsx_exists(scenario):
    matches = list(scenario["result_dir"].glob("*_richness_diversity.xlsx"))
    assert matches, "richness_diversity.xlsx não gerado"


def test_richness_diversity_has_data(scenario):
    df = load_xlsx(scenario["result_dir"], "richness_diversity")
    assert len(df) > 0, "richness_diversity.xlsx está vazio"
    assert "richness" in df.columns, "coluna 'richness' ausente"


def test_richness_values_positive(scenario):
    df = load_xlsx(scenario["result_dir"], "richness_diversity")
    assert (df["richness"] > 0).all(), "richness contém zeros ou negativos"


def test_shannon_diversity_non_negative(scenario):
    df = load_xlsx(scenario["result_dir"], "richness_diversity")
    if "shannon" not in df.columns:
        pytest.skip("coluna shannon ausente")
    assert (df["shannon"] >= 0).all(), "Shannon contém valores negativos"


# ── BLOCO 5: Lipinski e açúcares ──────────────────────────────────────────────

def test_lipinski_summary_exists(scenario):
    matches = list(scenario["result_dir"].glob("*_lipinski_sugars_summary.xlsx"))
    assert matches, "lipinski_sugars_summary.xlsx não gerado"


def test_lipinski_pct_ok_between_0_and_1(scenario):
    df = load_xlsx(scenario["result_dir"], "lipinski_sugars_summary")
    if "pct_Lipinski_OK" not in df.columns:
        pytest.skip("coluna pct_Lipinski_OK ausente")
    vals = df["pct_Lipinski_OK"].dropna()
    assert ((vals >= 0) & (vals <= 1)).all(), "pct_Lipinski_OK fora de [0, 1]"


def test_pct_sugar_between_0_and_1(scenario):
    df = load_xlsx(scenario["result_dir"], "lipinski_sugars_summary")
    if "pct_with_sugar" not in df.columns:
        pytest.skip("coluna pct_with_sugar ausente")
    vals = df["pct_with_sugar"].dropna()
    assert ((vals >= 0) & (vals <= 1)).all(), "pct_with_sugar fora de [0, 1]"


# ── BLOCO 6: Nível taxonômico correto no output ───────────────────────────────

def test_analysis_level_column_present_in_richness(scenario):
    """Coluna do nível de análise (genus/species) deve existir na tabela de riqueza."""
    cfg = scenario["config"]
    level = cfg["analysis_tax_level"]
    df = load_xlsx(scenario["result_dir"], "richness_diversity")
    col = next(
        (c for c in df.columns if c.lower() in (level, "taxon")), None
    )
    assert col is not None, (
        f"Coluna de nível '{level}' ou 'taxon' ausente em richness_diversity. "
        f"Colunas: {list(df.columns)}"
    )


def test_family_filter_respected(scenario):
    """lin_enriched deve conter apenas a(s) família(s) alvo quando mode=family."""
    cfg = scenario["config"]
    if cfg["taxon_mode"] != "family":
        pytest.skip("Cenário não é mode=family")
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    families_in_data = set(lin["family"].dropna().unique())
    target = set(cfg["taxon_values"])
    unexpected = families_in_data - target
    assert not unexpected, f"Famílias inesperadas em lin_enriched: {unexpected}"


def test_genus_filter_respected(scenario):
    """lin_enriched deve conter apenas o(s) gênero(s) alvo quando mode=genus."""
    cfg = scenario["config"]
    if cfg["taxon_mode"] != "genus":
        pytest.skip("Cenário não é mode=genus")
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    genera_in_data = set(lin["genus"].dropna().unique())
    target = set(cfg["taxon_values"])
    unexpected = genera_in_data - target
    assert not unexpected, f"Gêneros inesperados em lin_enriched: {unexpected}"


# ── BLOCO 7: Estatísticas de impacto (impressas, não falham) ──────────────────

def test_impact_summary(scenario, capsys):
    """Imprime métricas-chave para comparação de impacto entre runs."""
    cfg = scenario["config"]
    d = scenario["result_dir"]

    lin = load_parquet(d, "lin_enriched")
    uni = load_parquet(d, "uni_enriched")

    level = cfg["analysis_tax_level"]
    level_col = level if level in lin.columns else "genus"

    n_compounds_lin = len(lin)
    n_unique_inchi = lin["inchikey"].nunique()
    n_taxa = lin[level_col].nunique() if level_col in lin.columns else None

    mw = pd.to_numeric(uni["molecular_weight"], errors="coerce").dropna()
    xlogp = pd.to_numeric(uni.get("xlogp", pd.Series(dtype=float)), errors="coerce").dropna()

    with capsys.disabled():
        print(f"\n{'-'*55}")
        print(f"  IMPACTO - {cfg['id']}")
        print(f"{'-'*55}")
        print(f"  Diretório      : {d.name}")
        print(f"  lin_enriched   : {n_compounds_lin:,} linhas | {n_unique_inchi:,} InChIKeys únicos")
        print(f"  uni_enriched   : {len(uni):,} compostos únicos")
        print(f"  Táxons ({level_col:<8}): {n_taxa}")
        print(f"  MW  médio      : {mw.mean():.1f} Da  (±{mw.std():.1f})")
        print(f"  xLogP médio    : {xlogp.mean():.2f}  (±{xlogp.std():.2f})" if len(xlogp) else "  xLogP         : N/A")
        print(f"{'-'*55}")

    assert True  # este teste sempre passa; serve apenas para exibir métricas


# ── BLOCO A: Part I — lin_compound_species ────────────────────────────────────

def test_lin_compound_species_not_empty(scenario):
    df = load_parquet(scenario["result_dir"], "lin_compound_species")
    assert len(df) > 0, "lin_compound_species está vazio"


def test_lin_compound_species_required_columns(scenario):
    df = load_parquet(scenario["result_dir"], "lin_compound_species")
    required = ["inchikey", "species", "family", "genus"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em lin_compound_species: {missing}"


def test_lin_compound_species_subset_of_lin_enriched(scenario):
    """Todo inchikey em lin_compound_species deve existir em lin_enriched."""
    lcs = load_parquet(scenario["result_dir"], "lin_compound_species")
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    extra = set(lcs["inchikey"].dropna()) - set(lin["inchikey"].dropna())
    assert not extra, f"{len(extra)} inchikeys em lin_compound_species ausentes em lin_enriched"


# ── BLOCO B: Part III — estatísticas (XLSX) ───────────────────────────────────

def test_shared_compounds_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "shared_compounds")
    required = ["inchikey", "n_taxa", "taxa"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em shared_compounds: {missing}"


def test_shared_compounds_n_taxa_positive(scenario):
    df = load_xlsx(scenario["result_dir"], "shared_compounds")
    assert (df["n_taxa"] >= 1).all(), "shared_compounds tem n_taxa < 1"


def test_bibliometrics_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "bibliometrics")
    required = ["n_compounds", "pct_with_reference"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em bibliometrics: {missing}"


def test_bibliometrics_pct_reference_range(scenario):
    df = load_xlsx(scenario["result_dir"], "bibliometrics")
    vals = pd.to_numeric(df["pct_with_reference"], errors="coerce").dropna()
    assert ((vals >= 0) & (vals <= 1)).all(), "pct_with_reference fora de [0, 1]"


def test_pcoa_coordinates_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "PCoA_Coordinates")
    required = ["PCoA1", "PCoA2"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em PCoA_Coordinates: {missing}"


def test_pcoa_coordinates_finite(scenario):
    df = load_xlsx(scenario["result_dir"], "PCoA_Coordinates")
    for col in ["PCoA1", "PCoA2"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        n_inf = (~vals.isna() & ~vals.apply(lambda x: pd.notna(x) and abs(x) < 1e15)).sum()
        assert n_inf == 0, f"PCoA_Coordinates.{col} contém valores não finitos"


def test_stats_physchem_sheets(scenario):
    matches = list(scenario["result_dir"].glob("*_STATS_PhysChem*.xlsx"))
    if not matches:
        pytest.skip("STATS_PhysChem.xlsx não encontrado")
    xl = pd.ExcelFile(matches[0])
    assert "Descriptive_Global" in xl.sheet_names, "Aba 'Descriptive_Global' ausente"


def test_stats_physchem_required_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "STATS_PhysChem")
    required = ["Variable", "Group", "Mean", "Median", "SD", "N"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em STATS_PhysChem: {missing}"


def test_stats_physchem_n_positive(scenario):
    df = load_xlsx(scenario["result_dir"], "STATS_PhysChem")
    n_vals = pd.to_numeric(df["N"], errors="coerce").dropna()
    assert (n_vals > 0).all(), "STATS_PhysChem tem N <= 0"


def test_stats_chem_enrichment_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "STATS_Chem_Enrichment")
    required = ["Taxon", "Chemical_Class", "Odds_Ratio", "P_Value", "FDR_Adj_P"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em STATS_Chem_Enrichment: {missing}"


def test_stats_chem_enrichment_pvalue_range(scenario):
    df = load_xlsx(scenario["result_dir"], "STATS_Chem_Enrichment")
    for col in ["P_Value", "FDR_Adj_P"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), f"{col} fora de [0, 1]"


def test_stats_scaffold_innovation_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "STATS_Scaffold_Innovation")
    required = ["Taxon", "N_Compounds", "N_Scaffolds", "Innovation_Ratio"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em STATS_Scaffold_Innovation: {missing}"


def test_stats_scaffold_innovation_ratio_range(scenario):
    df = load_xlsx(scenario["result_dir"], "STATS_Scaffold_Innovation")
    vals = pd.to_numeric(df["Innovation_Ratio"], errors="coerce").dropna()
    assert ((vals >= 0) & (vals <= 1)).all(), "Innovation_Ratio fora de [0, 1]"


# ── BLOCO C: Part IV — bioatividade ───────────────────────────────────────────

def test_bio_a_summary_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "BIO_A_Summary")
    required = ["inchikey", "N_Assays", "Evidence_Flag_A"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em BIO_A_Summary: {missing}"


def test_bio_a_summary_n_assays_non_negative(scenario):
    df = load_xlsx(scenario["result_dir"], "BIO_A_Summary")
    vals = pd.to_numeric(df["N_Assays"], errors="coerce").dropna()
    assert (vals >= 0).all(), "BIO_A_Summary tem N_Assays negativo"


def test_bio_c_global_context_columns(scenario):
    df = load_xlsx(scenario["result_dir"], "BIO_C_Global_Context")
    required = ["inchikey", "Global_Family_Count", "Biogeography_Status"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em BIO_C_Global_Context: {missing}"


def test_bio_c_global_family_count_non_negative(scenario):
    """Global_Family_Count deve ser >= 0; compostos sem match global ficam em 0."""
    df = load_xlsx(scenario["result_dir"], "BIO_C_Global_Context")
    vals = pd.to_numeric(df["Global_Family_Count"], errors="coerce").dropna()
    assert (vals >= 0).all(), "Global_Family_Count contém valores negativos"
    pct_zero = (vals == 0).mean() * 100
    assert pct_zero < 10, f"{pct_zero:.1f}% dos compostos com Global_Family_Count=0 (limite: 10%)"


def test_master_list_sheets(scenario):
    matches = list(scenario["result_dir"].glob("*_MASTER_LIST_GLOBAL*.xlsx"))
    if not matches:
        pytest.skip("MASTER_LIST_GLOBAL.xlsx não encontrado")
    xl = pd.ExcelFile(matches[0])
    for sheet in ["All_Ranked", "Top_STARS", "Top_GEMS"]:
        assert sheet in xl.sheet_names, f"Aba '{sheet}' ausente em MASTER_LIST_GLOBAL"


def test_master_list_all_ranked_not_empty(scenario):
    matches = list(scenario["result_dir"].glob("*_MASTER_LIST_GLOBAL*.xlsx"))
    if not matches:
        pytest.skip("MASTER_LIST_GLOBAL.xlsx não encontrado")
    df = pd.read_excel(matches[0], sheet_name="All_Ranked")
    assert len(df) > 0, "MASTER_LIST_GLOBAL All_Ranked está vazio"


def test_master_list_required_columns(scenario):
    matches = list(scenario["result_dir"].glob("*_MASTER_LIST_GLOBAL*.xlsx"))
    if not matches:
        pytest.skip("MASTER_LIST_GLOBAL.xlsx não encontrado")
    df = pd.read_excel(matches[0], sheet_name="All_Ranked")
    required = ["inchikey", "molecular_weight"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colunas ausentes em MASTER_LIST_GLOBAL All_Ranked: {missing}"


def test_top25_discussion_sheets(scenario):
    matches = list(scenario["result_dir"].glob("*_Top25_Discussion_FullData*.xlsx"))
    if not matches:
        pytest.skip("Top25_Discussion_FullData.xlsx não encontrado")
    xl = pd.ExcelFile(matches[0])
    for sheet in ["Top25_STARS", "Top25_GEMS"]:
        assert sheet in xl.sheet_names, f"Aba '{sheet}' ausente em Top25_Discussion"


def test_top25_priority_score_present(scenario):
    matches = list(scenario["result_dir"].glob("*_Top25_Discussion_FullData*.xlsx"))
    if not matches:
        pytest.skip("Top25_Discussion_FullData.xlsx não encontrado")
    df = pd.read_excel(matches[0], sheet_name="Top25_STARS")
    assert "PRIORITY_SCORE" in df.columns, "Coluna PRIORITY_SCORE ausente em Top25_STARS"


def test_top25_inchikeys_in_master_list(scenario):
    """Todo inchikey do Top25 deve existir no MASTER_LIST_GLOBAL."""
    top25_files = list(scenario["result_dir"].glob("*_Top25_Discussion_FullData*.xlsx"))
    master_files = list(scenario["result_dir"].glob("*_MASTER_LIST_GLOBAL*.xlsx"))
    if not top25_files or not master_files:
        pytest.skip("Arquivo Top25 ou MASTER_LIST ausente")
    top25 = pd.read_excel(top25_files[0], sheet_name="Top25_STARS")
    master = pd.read_excel(master_files[0], sheet_name="All_Ranked")
    top_keys = set(top25["inchikey"].dropna())
    master_keys = set(master["inchikey"].dropna())
    missing = top_keys - master_keys
    assert not missing, f"{len(missing)} inchikeys do Top25 ausentes em MASTER_LIST"


# ── BLOCO 8: Snapshot / baseline — detecta regressões numéricas ───────────────
#
# Valores capturados em 2026-06-08 (resultados de referência).
# Tolerâncias: ±2 % para contagens, ±3 % para médias físico-químicas.
# Se um teste falhar aqui, significa que o código mudou o output de forma
# mensurável — verifique se a mudança era intencional e atualize o SNAPSHOT.

SNAPSHOT: dict[str, dict] = {
    "family_Phyllanthaceae_genus": {
        "lin_rows":        2054,
        "lin_unique_inchi": 1261,
        "uni_rows":        1261,
        "rd_rows":           12,   # número de gêneros na tabela de riqueza
        "lip_rows":          12,
        "mw_mean":        470.40,
        "xlogp_mean":       3.32,
        "fsp3_mean":        0.550,
    },
    "family_Melastomataceae_genus": {
        "lin_rows":         685,
        "lin_unique_inchi":  332,
        "uni_rows":          332,
        "rd_rows":            11,
        "lip_rows":           11,
        "mw_mean":         744.31,
        "xlogp_mean":        5.13,
        "fsp3_mean":         0.343,
    },
    "genus_Duguetia_species": {
        "lin_rows":         145,
        "lin_unique_inchi":  125,
        "uni_rows":          125,
        "rd_rows":             5,
        "lip_rows":            5,
        "mw_mean":         282.90,
        "xlogp_mean":        3.95,
        "fsp3_mean":         0.532,
    },
}

_COUNT_TOL = 0.02   # ±2 % para contagens de linhas / inchikeys
_STAT_TOL  = 0.03   # ±3 % para médias físico-químicas


def _snap(scenario_id: str) -> dict:
    snap = SNAPSHOT.get(scenario_id)
    if snap is None:
        pytest.skip(f"Sem snapshot definido para '{scenario_id}'")
    return snap


def test_snapshot_lin_row_count(scenario):
    """lin_enriched não deve ganhar nem perder mais de 2 % das linhas."""
    snap = _snap(scenario["config"]["id"])
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    expected = snap["lin_rows"]
    pct = abs(len(lin) - expected) / expected
    assert pct <= _COUNT_TOL, (
        f"lin_enriched: {len(lin)} linhas (esperado {expected} ±{_COUNT_TOL*100:.0f}%)"
    )


def test_snapshot_lin_unique_inchikeys(scenario):
    """Número de InChIKeys únicos em lin_enriched não deve variar >2 %."""
    snap = _snap(scenario["config"]["id"])
    lin = load_parquet(scenario["result_dir"], "lin_enriched")
    got = lin["inchikey"].nunique()
    expected = snap["lin_unique_inchi"]
    pct = abs(got - expected) / expected
    assert pct <= _COUNT_TOL, (
        f"lin_enriched unique inchikeys: {got} (esperado {expected} ±{_COUNT_TOL*100:.0f}%)"
    )


def test_snapshot_uni_row_count(scenario):
    """uni_enriched não deve ganhar nem perder mais de 2 % dos compostos."""
    snap = _snap(scenario["config"]["id"])
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    expected = snap["uni_rows"]
    pct = abs(len(uni) - expected) / expected
    assert pct <= _COUNT_TOL, (
        f"uni_enriched: {len(uni)} linhas (esperado {expected} ±{_COUNT_TOL*100:.0f}%)"
    )


def test_snapshot_richness_taxon_count(scenario):
    """Número de táxons na tabela de riqueza deve ser exato (filtro determinístico)."""
    snap = _snap(scenario["config"]["id"])
    df = load_xlsx(scenario["result_dir"], "richness_diversity")
    assert len(df) == snap["rd_rows"], (
        f"richness_diversity: {len(df)} táxons (esperado exato: {snap['rd_rows']})"
    )


def test_snapshot_mw_mean(scenario):
    """MW médio não deve desviar >3 % do baseline."""
    snap = _snap(scenario["config"]["id"])
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    mw = pd.to_numeric(uni["molecular_weight"], errors="coerce").dropna()
    got = mw.mean()
    expected = snap["mw_mean"]
    pct = abs(got - expected) / expected
    assert pct <= _STAT_TOL, (
        f"MW médio: {got:.2f} Da (esperado {expected:.2f} ±{_STAT_TOL*100:.0f}%)"
    )


def test_snapshot_xlogp_mean(scenario):
    """xLogP médio não deve desviar >3 % do baseline."""
    snap = _snap(scenario["config"]["id"])
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    if "xlogp" not in uni.columns:
        pytest.skip("coluna xlogp ausente")
    xlp = pd.to_numeric(uni["xlogp"], errors="coerce").dropna()
    got = xlp.mean()
    expected = snap["xlogp_mean"]
    pct = abs(got - expected) / abs(expected)
    assert pct <= _STAT_TOL, (
        f"xLogP médio: {got:.3f} (esperado {expected:.3f} ±{_STAT_TOL*100:.0f}%)"
    )


def test_snapshot_fsp3_mean(scenario):
    """Fsp3 médio não deve desviar >3 % do baseline."""
    snap = _snap(scenario["config"]["id"])
    uni = load_parquet(scenario["result_dir"], "uni_enriched")
    if "fsp3" not in uni.columns:
        pytest.skip("coluna fsp3 ausente")
    fsp = pd.to_numeric(uni["fsp3"], errors="coerce").dropna()
    got = fsp.mean()
    expected = snap["fsp3_mean"]
    pct = abs(got - expected) / expected
    assert pct <= _STAT_TOL, (
        f"Fsp3 médio: {got:.4f} (esperado {expected:.4f} ±{_STAT_TOL*100:.0f}%)"
    )

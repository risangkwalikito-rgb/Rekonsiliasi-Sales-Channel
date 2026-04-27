# app.py

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_PORTS = [
    "MERAK",
    "BAKAUHENI",
    "KETAPANG",
    "GILIMANUK",
    "CIWANDAN",
    "PANJANG",
    "WIKA BETON",
]

PORT_ALIASES = {
    "MERAK": ["MERAK"],
    "BAKAUHENI": ["BAKAUHENI"],
    "KETAPANG": ["KETAPANG"],
    "GILIMANUK": ["GILIMANUK"],
    "CIWANDAN": ["CIWANDAN"],
    "PANJANG": ["PANJANG"],
    "WIKA BETON": ["WIKA BETON", "WIKA", "WIKA_BETON"],
}

DATE_CANDIDATES = [
    "CETAK BOARDING PASS",
    "BOARDING PASS",
    "TGL CETAK BOARDING PASS",
    "TANGGAL",
    "TGL",
    "TGL INVOICE",
    "INVOICE DATE",
    "TGL TRANSAKSI",
    "TRANSACTION DATE",
    "DATE",
    "CREATED AT",
]

PORT_CANDIDATES = [
    "ASAL",
    "PELABUHAN",
    "CABANG",
    "PELABUHAN (ASAL)",
    "CABANG / PELABUHAN",
    "CABANG/PELABUHAN",
    "PORT",
    "ORIGIN",
]

AMOUNT_CANDIDATES_SUMMARY = [
    "TARIF",
    "NOMINAL",
    "TOTAL",
    "AMOUNT",
    "JUMLAH",
]

AMOUNT_CANDIDATES_INVOICE = [
    "GRAND TOTAL",
    "TOTAL",
    "NOMINAL",
    "AMOUNT",
    "TOTAL TAGIHAN",
    "TAGIHAN",
    "DPP",
    "TARIF",
    "JUMLAH",
]

AMOUNT_CANDIDATES_TICKET = [
    "SUB TOTAL",
    "SUBTOTAL",
    "TOTAL JUMLAH",
    "GRAND TOTAL",
    "TOTAL",
    "NOMINAL",
    "AMOUNT",
    "JUMLAH",
]


@dataclass
class AggregateResult:
    series: pd.Series
    detail: pd.DataFrame
    warnings: list[str]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_header(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("\n", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_money_text(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = cleaned.replace("Rp", "").replace("rp", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = re.sub(r"[^\d,.\-]", "", cleaned)
    return cleaned


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)

    text = clean_money_text(str(value))
    if not text or text in {"-", ".", ","}:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    elif text.count(",") > 1:
        text = text.replace(",", "")
    elif "," in text and "." not in text:
        parts = text.split(",")
        if len(parts[-1]) in {1, 2}:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def parse_last_number_from_row(values: Iterable[Any]) -> float | None:
    numeric_values: list[float] = []
    for value in values:
        number = parse_number(value)
        if number is not None:
            numeric_values.append(number)
    if numeric_values:
        return numeric_values[-1]

    row_text = " ".join(str(v) for v in values if v is not None)
    matches = re.findall(r"-?\d[\d.,]*", row_text)
    if not matches:
        return None
    return parse_number(matches[-1])


def canonical_port(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    for canonical, aliases in PORT_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return canonical
    return None


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_header(col) for col in result.columns]
    result = result.dropna(axis=0, how="all").dropna(axis=1, how="all")
    result = result.reset_index(drop=True)
    return result


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_columns = [normalize_header(col) for col in columns]
    normalized_candidates = [normalize_header(c) for c in candidates]

    for candidate in normalized_candidates:
        for column in normalized_columns:
            if column == candidate:
                return column

    for candidate in normalized_candidates:
        for column in normalized_columns:
            if candidate in column:
                return column

    return None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.notna().sum() > 0:
        return parsed

    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y %H:%M:%S"]
    text_series = series.astype(str)
    for fmt in formats:
        parsed = pd.to_datetime(text_series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed
    return pd.to_datetime(series.astype(str), errors="coerce")


def exact_date_mask(series: pd.Series, target_date: date) -> pd.Series:
    parsed = parse_datetime_series(series)
    if parsed.notna().sum() > 0:
        return parsed.dt.date == target_date

    text = series.astype(str).str.upper()
    patterns = [
        target_date.strftime("%Y-%m-%d"),
        target_date.strftime("%d/%m/%Y"),
        target_date.strftime("%d-%m-%Y"),
        target_date.strftime("%m/%d/%Y"),
        target_date.strftime("%d %m %Y"),
    ]
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask = mask | text.str.contains(pattern, regex=False, na=False)
    return mask


def between_date_mask(series: pd.Series, start_date: date, end_date: date) -> pd.Series:
    parsed = parse_datetime_series(series)
    if parsed.notna().sum() == 0:
        return pd.Series(True, index=series.index)
    return (parsed.dt.date >= start_date) & (parsed.dt.date <= end_date)


def read_uploaded_file(uploaded_file: Any) -> dict[str, pd.DataFrame]:
    file_name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    buffer = io.BytesIO(content)

    if file_name.endswith((".xlsx", ".xlsm", ".xls")):
        raw = pd.read_excel(buffer, sheet_name=None, dtype=object)
        return {sheet_name: prepare_dataframe(df) for sheet_name, df in raw.items()}

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(buffer, dtype=object)
        except Exception:
            buffer.seek(0)
            df = pd.read_csv(buffer, dtype=object, sep=";")
        return {"CSV": prepare_dataframe(df)}

    raise ValueError("Format file belum didukung. Gunakan Excel atau CSV.")


def combine_selected_sheets(sheet_map: dict[str, pd.DataFrame], selected_sheets: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sheet_name in selected_sheets:
        df = sheet_map[sheet_name].copy()
        df["__SHEET__"] = sheet_name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def base_result() -> pd.Series:
    return pd.Series(0.0, index=DEFAULT_PORTS, dtype=float)


def aggregate_summary_exact_date(df: pd.DataFrame, target_date: date) -> AggregateResult:
    warnings: list[str] = []
    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Summary kosong."])

    date_col = detect_column(df.columns.tolist(), ["CETAK BOARDING PASS"] + DATE_CANDIDATES)
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_SUMMARY)

    if date_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom tanggal Tiket Summary tidak ditemukan."])
    if port_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom ASAL/Pelabuhan Tiket Summary tidak ditemukan."])
    if amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom Tarif/Nominal Tiket Summary tidak ditemukan."])

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work = work[work["__PORT__"].notna() & work["__AMOUNT__"].notna()].copy()

    mask = exact_date_mask(work[date_col], target_date)
    work = work[mask].copy()

    grouped = work.groupby("__PORT__")["__AMOUNT__"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = work[[date_col, port_col, amount_col, "__PORT__", "__AMOUNT__", "__SHEET__"]].rename(
        columns={
            date_col: "TANGGAL_SUMMARY",
            port_col: "ASAL_SUMBER",
            amount_col: "NOMINAL_SUMBER",
            "__PORT__": "PELABUHAN",
            "__AMOUNT__": "NOMINAL",
            "__SHEET__": "SHEET",
        }
    )
    if detail.empty:
        warnings.append(f"Tidak ada data Tiket Summary untuk tanggal {target_date.strftime('%d-%m-%Y')}.")

    return AggregateResult(result, detail, warnings)


def aggregate_summary_range(df: pd.DataFrame, start_date: date, end_date: date) -> AggregateResult:
    warnings: list[str] = []
    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Summary kosong."])

    date_col = detect_column(df.columns.tolist(), ["CETAK BOARDING PASS"] + DATE_CANDIDATES)
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_SUMMARY)

    if date_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom tanggal Tiket Summary tidak ditemukan."])
    if port_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom ASAL/Pelabuhan Tiket Summary tidak ditemukan."])
    if amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom Tarif/Nominal Tiket Summary tidak ditemukan."])

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work = work[work["__PORT__"].notna() & work["__AMOUNT__"].notna()].copy()

    mask = between_date_mask(work[date_col], start_date, end_date)
    work = work[mask].copy()

    grouped = work.groupby("__PORT__")["__AMOUNT__"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = work[[date_col, port_col, amount_col, "__PORT__", "__AMOUNT__", "__SHEET__"]].rename(
        columns={
            date_col: "TANGGAL_SUMMARY",
            port_col: "ASAL_SUMBER",
            amount_col: "NOMINAL_SUMBER",
            "__PORT__": "PELABUHAN",
            "__AMOUNT__": "NOMINAL",
            "__SHEET__": "SHEET",
        }
    )
    if detail.empty:
        warnings.append(
            f"Tidak ada data Tiket Summary untuk rentang {start_date.strftime('%d-%m-%Y')} s.d. {end_date.strftime('%d-%m-%Y')}."
        )

    return AggregateResult(result, detail, warnings)


def aggregate_invoice_range(df: pd.DataFrame, start_date: date, end_date: date) -> AggregateResult:
    warnings: list[str] = []
    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Invoice kosong."])

    date_col = detect_column(df.columns.tolist(), DATE_CANDIDATES)
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_INVOICE)

    if port_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom ASAL/Pelabuhan Invoice tidak ditemukan."])
    if amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom Nominal/Total Invoice tidak ditemukan."])

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work = work[work["__PORT__"].notna() & work["__AMOUNT__"].notna()].copy()

    if date_col is not None:
        mask = between_date_mask(work[date_col], start_date, end_date)
        work = work[mask].copy()
    else:
        warnings.append("Kolom tanggal Invoice tidak ditemukan. Perhitungan Invoice memakai seluruh baris yang cocok.")

    grouped = work.groupby("__PORT__")["__AMOUNT__"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    selected_columns = [port_col, amount_col, "__PORT__", "__AMOUNT__", "__SHEET__"]
    rename_map = {
        port_col: "ASAL_SUMBER",
        amount_col: "NOMINAL_SUMBER",
        "__PORT__": "PELABUHAN",
        "__AMOUNT__": "NOMINAL",
        "__SHEET__": "SHEET",
    }
    if date_col is not None:
        selected_columns.insert(0, date_col)
        rename_map[date_col] = "TANGGAL_INVOICE"

    detail = work[selected_columns].rename(columns=rename_map)
    if detail.empty:
        warnings.append(
            f"Tidak ada data Invoice untuk rentang {start_date.strftime('%d-%m-%Y')} s.d. {end_date.strftime('%d-%m-%Y')}."
        )

    return AggregateResult(result, detail, warnings)


def extract_ticket_sold_totals(df: pd.DataFrame) -> AggregateResult:
    warnings: list[str] = []
    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Terjual kosong."])

    structured = extract_ticket_sold_structured(df)
    if structured.series.sum() > 0:
        return structured

    fallback = extract_ticket_sold_report_style(df)
    if fallback.series.sum() == 0:
        warnings.extend(structured.warnings)
        warnings.extend(fallback.warnings)
    return AggregateResult(fallback.series, fallback.detail, warnings or fallback.warnings)


def extract_ticket_sold_structured(df: pd.DataFrame) -> AggregateResult:
    warnings: list[str] = []
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_TICKET)

    if port_col is None or amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Mode kolom terstruktur Tiket Terjual tidak terdeteksi."])

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__ROW_TEXT__"] = work.astype(str).agg(" | ".join, axis=1).apply(normalize_text)
    work = work[work["__PORT__"].notna() & work["__AMOUNT__"].notna()].copy()

    if work.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["Tidak ada data Tiket Terjual yang cocok pada mode terstruktur."])

    subtotal_mask = (
        work["__ROW_TEXT__"].str.contains("SUB TOTAL", regex=False, na=False)
        | work["__ROW_TEXT__"].str.contains("SUBTOTAL", regex=False, na=False)
        | work["__ROW_TEXT__"].str.contains("TOTAL JUMLAH", regex=False, na=False)
    )

    if subtotal_mask.any():
        subtotal_rows = work[subtotal_mask].copy()
        subtotal_rows["__ORDER__"] = np.arange(len(subtotal_rows))
        subtotal_rows = subtotal_rows.sort_values(["__PORT__", "__ORDER__"])
        chosen = subtotal_rows.groupby("__PORT__", as_index=False).tail(1)

        grouped = chosen.groupby("__PORT__")["__AMOUNT__"].sum()
        detail_source = chosen
    else:
        grouped = work.groupby("__PORT__")["__AMOUNT__"].sum()
        detail_source = work

    result = base_result().add(grouped, fill_value=0.0)

    detail = detail_source[[port_col, amount_col, "__PORT__", "__AMOUNT__", "__SHEET__"]].rename(
        columns={
            port_col: "SUMBER_PORT",
            amount_col: "NOMINAL_SUMBER",
            "__PORT__": "PELABUHAN",
            "__AMOUNT__": "NOMINAL",
            "__SHEET__": "SHEET",
        }
    )
    return AggregateResult(result, detail, warnings)


def extract_ticket_sold_report_style(df: pd.DataFrame) -> AggregateResult:
    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    current_port: str | None = None

    for idx, row in df.iterrows():
        row_values = row.tolist()
        row_text = " | ".join("" if pd.isna(v) else str(v) for v in row_values)
        normalized_row_text = normalize_text(row_text)
        explicit_port = canonical_port(normalized_row_text)
        if explicit_port:
            current_port = explicit_port

        target_port = explicit_port or current_port
        if target_port is None:
            continue

        amount = parse_last_number_from_row(row_values)
        if amount is None:
            continue

        priority = 0
        if "SUB TOTAL" in normalized_row_text or "SUBTOTAL" in normalized_row_text:
            priority = 3
        elif "TOTAL JUMLAH" in normalized_row_text:
            priority = 2
        elif explicit_port is not None:
            priority = 1

        if "GRAND TOTAL" in normalized_row_text and explicit_port is None and priority == 0:
            continue
        if priority == 0:
            continue

        candidates.append(
            {
                "ROW_INDEX": idx,
                "PELABUHAN": target_port,
                "NOMINAL": float(amount),
                "PRIORITY": priority,
                "ROW_TEXT": row_text,
                "SHEET": row.get("__SHEET__", ""),
            }
        )

    if not candidates:
        return AggregateResult(base_result(), pd.DataFrame(), ["Parser subtotal/total Tiket Terjual tidak menemukan baris yang cocok."])

    candidate_df = pd.DataFrame(candidates)
    candidate_df = candidate_df.sort_values(["PELABUHAN", "PRIORITY", "ROW_INDEX"], ascending=[True, False, False])

    chosen_rows: list[pd.Series] = []
    for port in DEFAULT_PORTS:
        subset = candidate_df[candidate_df["PELABUHAN"] == port]
        if subset.empty:
            continue
        chosen_rows.append(subset.iloc[0])

    if not chosen_rows:
        return AggregateResult(base_result(), pd.DataFrame(), ["Tidak ada subtotal/total Tiket Terjual yang cocok per pelabuhan."])

    chosen_df = pd.DataFrame(chosen_rows)
    grouped = chosen_df.groupby("PELABUHAN")["NOMINAL"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = chosen_df[["PELABUHAN", "NOMINAL", "PRIORITY", "ROW_INDEX", "ROW_TEXT", "SHEET"]].rename(
        columns={"ROW_TEXT": "BARIS_SUMBER"}
    )
    return AggregateResult(result, detail, warnings)


def format_currency_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(
                lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else ""
            )
    return formatted


def build_reconciliation(
    ticket_sold_df: pd.DataFrame,
    ticket_summary_df: pd.DataFrame,
    invoice_df: pd.DataFrame,
    summary_start_date: date,
    summary_end_date: date,
    ntg_start_date: date,
    ntg_end_date: date,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    warnings: list[str] = []

    ticket_sold = extract_ticket_sold_totals(ticket_sold_df)
    addition = aggregate_summary_exact_date(ticket_summary_df, summary_start_date)
    deduction = aggregate_summary_exact_date(ticket_summary_df, summary_end_date)
    summary_range = aggregate_summary_range(ticket_summary_df, ntg_start_date, ntg_end_date)
    invoice_range = aggregate_invoice_range(invoice_df, ntg_start_date, ntg_end_date)

    warnings.extend(ticket_sold.warnings)
    warnings.extend(addition.warnings)
    warnings.extend(deduction.warnings)
    warnings.extend(summary_range.warnings)
    warnings.extend(invoice_range.warnings)

    result = pd.DataFrame({"Pelabuhan (ASAL)": DEFAULT_PORTS})
    result["Nominal Tiket Terjual"] = result["Pelabuhan (ASAL)"].map(ticket_sold.series).fillna(0.0)
    result["Nominal Penambahan"] = result["Pelabuhan (ASAL)"].map(addition.series).fillna(0.0)
    result["Nominal Pengurangan"] = result["Pelabuhan (ASAL)"].map(deduction.series).fillna(0.0)
    result["Nominal Naik Turun Golongan"] = (
        result["Pelabuhan (ASAL)"].map(invoice_range.series).fillna(0.0)
        - result["Pelabuhan (ASAL)"].map(summary_range.series).fillna(0.0)
    )
    result["Nominal Pinbuk"] = (
        result["Nominal Tiket Terjual"]
        + result["Nominal Penambahan"]
        - result["Nominal Pengurangan"]
        + result["Nominal Naik Turun Golongan"]
    )

    detail_tables = {
        "tiket_terjual": ticket_sold.detail,
        "penambahan": addition.detail,
        "pengurangan": deduction.detail,
        "summary_range": summary_range.detail,
        "invoice_range": invoice_range.detail,
    }
    return result, detail_tables, warnings


def to_excel_bytes(reconciliation_df: pd.DataFrame, detail_tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        reconciliation_df.to_excel(writer, index=False, sheet_name="Rekonsiliasi")
        for sheet_name, detail_df in detail_tables.items():
            safe_sheet_name = sheet_name[:31]
            detail_df.to_excel(writer, index=False, sheet_name=safe_sheet_name)
    output.seek(0)
    return output.getvalue()


def file_section(label: str, key_prefix: str) -> tuple[dict[str, pd.DataFrame] | None, list[str]]:
    uploaded = st.file_uploader(
        label,
        type=["xlsx", "xls", "xlsm", "csv"],
        key=f"{key_prefix}_uploader",
    )
    if uploaded is None:
        return None, []

    try:
        sheet_map = read_uploaded_file(uploaded)
    except Exception as exc:
        st.error(f"Gagal membaca file {label}: {exc}")
        return None, []

    sheet_names = list(sheet_map.keys())
    selected = st.multiselect(
        f"Pilih sheet untuk {label}",
        options=sheet_names,
        default=sheet_names,
        key=f"{key_prefix}_sheets",
    )
    return sheet_map, selected


def render_preview(name: str, sheet_map: dict[str, pd.DataFrame] | None, selected_sheets: list[str]) -> pd.DataFrame:
    if not sheet_map or not selected_sheets:
        return pd.DataFrame()

    combined = combine_selected_sheets(sheet_map, selected_sheets)
    with st.expander(f"Preview {name}", expanded=False):
        st.caption(f"{len(selected_sheets)} sheet dipilih | {combined.shape[0]} baris | {combined.shape[1]} kolom")
        st.dataframe(combined.head(30), use_container_width=True, height=280)
        st.write("Kolom terdeteksi:", list(combined.columns))
    return combined


st.set_page_config(page_title="Rekonsiliasi Sales Channel", layout="wide")
st.title("Rekonsiliasi Sales Channel")
st.caption("Upload Tiket Terjual, Tiket Summary, dan Invoice untuk menghasilkan tabel rekonsiliasi per pelabuhan.")

with st.sidebar:
    st.subheader("Parameter")
    summary_start_date = st.date_input("Tanggal Penambahan (T-Summary awal)", value=date.today(), key="summary_start_date")
    summary_end_date = st.date_input("Tanggal Pengurangan (T-Summary akhir)", value=date.today(), key="summary_end_date")
    ntg_start_date = st.date_input("Tanggal Naik Turun Golongan - Mulai", value=date.today(), key="ntg_start_date")
    ntg_end_date = st.date_input("Tanggal Naik Turun Golongan - Selesai", value=date.today(), key="ntg_end_date")

    st.markdown("---")
    st.write("Pelabuhan default:")
    st.write(", ".join(DEFAULT_PORTS))

col1, col2, col3 = st.columns(3)
with col1:
    ticket_sold_map, ticket_sold_sheets = file_section("Uploader Tiket Terjual", "ticket_sold")
with col2:
    ticket_summary_map, ticket_summary_sheets = file_section("Uploader Tiket Summary", "ticket_summary")
with col3:
    invoice_map, invoice_sheets = file_section("Uploader Invoice", "invoice")

ticket_sold_df = render_preview("Tiket Terjual", ticket_sold_map, ticket_sold_sheets)
ticket_summary_df = render_preview("Tiket Summary", ticket_summary_map, ticket_summary_sheets)
invoice_df = render_preview("Invoice", invoice_map, invoice_sheets)

ready = all(
    [
        ticket_sold_map is not None,
        ticket_summary_map is not None,
        invoice_map is not None,
        len(ticket_sold_sheets) > 0,
        len(ticket_summary_sheets) > 0,
        len(invoice_sheets) > 0,
    ]
)

if not ready:
    st.info("Lengkapi 3 uploader dan pilih minimal 1 sheet pada masing-masing file.")
else:
    reconciliation_df, detail_tables, warnings_list = build_reconciliation(
        ticket_sold_df=ticket_sold_df,
        ticket_summary_df=ticket_summary_df,
        invoice_df=invoice_df,
        summary_start_date=summary_start_date,
        summary_end_date=summary_end_date,
        ntg_start_date=ntg_start_date,
        ntg_end_date=ntg_end_date,
    )

    numeric_columns = [
        "Nominal Tiket Terjual",
        "Nominal Penambahan",
        "Nominal Pengurangan",
        "Nominal Naik Turun Golongan",
        "Nominal Pinbuk",
    ]

    st.subheader("Tabel Rekonsiliasi Sales Channel")
    st.dataframe(format_currency_columns(reconciliation_df, numeric_columns), use_container_width=True, height=360)

    total_row = reconciliation_df[numeric_columns].sum().to_frame().T
    total_row.insert(0, "Pelabuhan (ASAL)", "TOTAL")
    st.subheader("Grand Total")
    st.dataframe(format_currency_columns(total_row, numeric_columns), use_container_width=True, hide_index=True)

    if warnings_list:
        unique_warnings = list(dict.fromkeys(warnings_list))
        with st.expander("Catatan parser", expanded=False):
            for warning in unique_warnings:
                st.warning(warning)

    excel_bytes = to_excel_bytes(reconciliation_df, detail_tables)
    st.download_button(
        label="Download Hasil Rekonsiliasi (Excel)",
        data=excel_bytes,
        file_name="rekonsiliasi_sales_channel.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tabs = st.tabs(
        [
            "Detail Tiket Terjual",
            "Detail Penambahan",
            "Detail Pengurangan",
            "Detail Summary Range",
            "Detail Invoice Range",
        ]
    )
    tab_keys = ["tiket_terjual", "penambahan", "pengurangan", "summary_range", "invoice_range"]

    for tab, key in zip(tabs, tab_keys):
        with tab:
            detail_df = detail_tables[key]
            if detail_df.empty:
                st.info("Tidak ada data detail.")
            else:
                display_df = detail_df.copy()
                amount_like_columns = [col for col in display_df.columns if "NOMINAL" in col.upper()]
                st.dataframe(format_currency_columns(display_df, amount_like_columns), use_container_width=True, height=420)

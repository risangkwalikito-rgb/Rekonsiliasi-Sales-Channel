# file: app.py

import io
import re
from dataclasses import dataclass
from datetime import date, datetime, time
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

INVOICE_NUMBER_CANDIDATES = [
    "NO INVOICE",
    "NOMOR INVOICE",
    "NO. INVOICE",
    "NO.INVOICE",
    "INVOICE NO",
    "INVOICE NUMBER",
    "NO INV",
    "NOMOR INV",
    "NO. INV",
    "INVOICE",
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

WINDOW_START = "00:00:00"
WINDOW_END = "07:59:59"


@dataclass
class AggregateResult:
    series: pd.Series
    detail: pd.DataFrame
    warnings: list[str]


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_header(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("\n", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_invoice_number(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip().upper()

    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    if text in {"", "NAN", "NONE", "-"}:
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def canonical_port(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None

    for canonical, aliases in PORT_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return canonical
    return None


def clean_money_text(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("Rp", "").replace("rp", "")
    text = text.replace(" ", "")
    text = re.sub(r"[^\d,.\-]", "", text)
    return text


def parse_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = clean_money_text(value)
    if text in {"", "-", ".", ","}:
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

    text = " ".join(str(v) for v in values if v is not None and not pd.isna(v))
    matches = re.findall(r"-?\d[\d.,]*", text)
    if not matches:
        return None
    return parse_number(matches[-1])


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


def detect_invoice_number_column(columns: list[str]) -> str | None:
    normalized_columns = [normalize_header(col) for col in columns]

    for candidate in [normalize_header(c) for c in INVOICE_NUMBER_CANDIDATES]:
        for column in normalized_columns:
            if column == candidate:
                return column

    for column in normalized_columns:
        if "INVOICE" in column and "DATE" not in column and "TGL" not in column:
            return column

    for column in normalized_columns:
        if "INV" in column and "DATE" not in column and "TGL" not in column:
            return column

    return None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.notna().sum() > 0:
        return parsed

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
    ]
    text_series = series.astype(str)

    for fmt in formats:
        parsed = pd.to_datetime(text_series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed

    return pd.to_datetime(text_series, errors="coerce")


def parse_time_value(value: str) -> time:
    return datetime.strptime(value, "%H:%M:%S").time()


def between_date_mask(series: pd.Series, start_date: date, end_date: date) -> pd.Series:
    parsed = parse_datetime_series(series)
    if parsed.notna().sum() == 0:
        return pd.Series(False, index=series.index)
    return (parsed.dt.date >= start_date) & (parsed.dt.date <= end_date)


def exact_date_time_window_mask(
    series: pd.Series,
    target_date: date,
    window_start: str = WINDOW_START,
    window_end: str = WINDOW_END,
) -> pd.Series:
    parsed = parse_datetime_series(series)
    if parsed.notna().sum() == 0:
        return pd.Series(False, index=series.index)

    start_time = parse_time_value(window_start)
    end_time = parse_time_value(window_end)

    return (parsed.dt.date == target_date) & (parsed.dt.time >= start_time) & (parsed.dt.time <= end_time)


def base_result() -> pd.Series:
    return pd.Series(0.0, index=DEFAULT_PORTS, dtype=float)


def read_uploaded_file(uploaded_file: Any) -> dict[str, pd.DataFrame]:
    file_name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    buffer = io.BytesIO(content)

    if file_name.endswith((".xlsx", ".xls", ".xlsm")):
        raw = pd.read_excel(buffer, sheet_name=None, dtype=object)
        return {sheet_name: prepare_dataframe(df) for sheet_name, df in raw.items()}

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(buffer, dtype=object)
        except Exception:
            buffer.seek(0)
            df = pd.read_csv(buffer, dtype=object, sep=";")
        return {"CSV": prepare_dataframe(df)}

    raise ValueError("Format file tidak didukung. Gunakan Excel atau CSV.")


def load_multiple_files(uploaded_files: list[Any]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    sheet_map: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            file_sheets = read_uploaded_file(uploaded_file)
            for sheet_name, df in file_sheets.items():
                key = f"{uploaded_file.name} :: {sheet_name}"
                df_copy = df.copy()
                df_copy["__FILE__"] = uploaded_file.name
                df_copy["__SHEET__"] = sheet_name
                sheet_map[key] = df_copy
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    return sheet_map, errors


def combine_selected_sheets(sheet_map: dict[str, pd.DataFrame], selected_keys: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for key in selected_keys:
        df = sheet_map[key].copy()
        df["__SOURCE__"] = key
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate_summary_window(df: pd.DataFrame, target_date: date) -> AggregateResult:
    warnings: list[str] = []

    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Summary kosong."])

    date_col = detect_column(df.columns.tolist(), ["CETAK BOARDING PASS"] + DATE_CANDIDATES)
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_SUMMARY)

    if date_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom CETAK BOARDING PASS/Tanggal Tiket Summary tidak ditemukan."])
    if port_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom ASAL/Pelabuhan Tiket Summary tidak ditemukan."])
    if amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Kolom Tarif/Nominal Tiket Summary tidak ditemukan."])

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__DATETIME__"] = parse_datetime_series(work[date_col])

    work = work[
        work["__PORT__"].notna()
        & work["__AMOUNT__"].notna()
        & work["__DATETIME__"].notna()
    ].copy()

    work = work[exact_date_time_window_mask(work[date_col], target_date, WINDOW_START, WINDOW_END)].copy()

    grouped = work.groupby("__PORT__")["__AMOUNT__"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = work[[date_col, port_col, amount_col, "__PORT__", "__AMOUNT__", "__SOURCE__"]].rename(
        columns={
            date_col: "CETAK_BOARDING_PASS",
            port_col: "ASAL_SUMBER",
            amount_col: "NOMINAL_SUMBER",
            "__PORT__": "PELABUHAN",
            "__AMOUNT__": "NOMINAL",
            "__SOURCE__": "SUMBER_FILE_SHEET",
        }
    )

    if detail.empty:
        warnings.append(
            f"Tidak ada data Tiket Summary untuk tanggal {target_date.strftime('%Y/%m/%d')} "
            f"jam {WINDOW_START} - {WINDOW_END}."
        )

    return AggregateResult(result, detail, warnings)


def aggregate_summary_by_invoice_number(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if df.empty:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), ["File Tiket Summary kosong."]

    date_col = detect_column(df.columns.tolist(), ["CETAK BOARDING PASS"] + DATE_CANDIDATES)
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_SUMMARY)
    invoice_col = detect_invoice_number_column(df.columns.tolist())

    if date_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), ["Kolom tanggal Tiket Summary tidak ditemukan."]
    if port_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), ["Kolom ASAL/Pelabuhan Tiket Summary tidak ditemukan."]
    if amount_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), ["Kolom Tarif/Nominal Tiket Summary tidak ditemukan."]
    if invoice_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), ["Kolom Nomor Invoice di Tiket Summary tidak ditemukan."]

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__INVOICE__"] = work[invoice_col].apply(normalize_invoice_number)
    work["__DATETIME__"] = parse_datetime_series(work[date_col])

    work = work[
        work["__PORT__"].notna()
        & work["__AMOUNT__"].notna()
        & work["__INVOICE__"].notna()
        & work["__DATETIME__"].notna()
    ].copy()

    work = work[between_date_mask(work[date_col], start_date, end_date)].copy()

    if work.empty:
        warnings.append(
            f"Tidak ada data Tiket Summary per nomor invoice untuk rentang "
            f"{start_date.strftime('%d-%m-%Y')} s.d. {end_date.strftime('%d-%m-%Y')}."
        )
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), warnings

    grouped = (
        work.groupby(["__PORT__", "__INVOICE__"], as_index=False)["__AMOUNT__"]
        .sum()
        .rename(
            columns={
                "__PORT__": "PELABUHAN",
                "__INVOICE__": "NOMOR_INVOICE",
                "__AMOUNT__": "NOMINAL_TIKET_SUMMARY",
            }
        )
    )

    return grouped, warnings


def aggregate_invoice_by_invoice_number(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if df.empty:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), ["File Invoice kosong."]

    date_col = detect_column(df.columns.tolist(), DATE_CANDIDATES)
    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_INVOICE)
    invoice_col = detect_invoice_number_column(df.columns.tolist())

    if port_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), ["Kolom ASAL/Pelabuhan di Invoice tidak ditemukan."]
    if amount_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), ["Kolom Nominal/Total di Invoice tidak ditemukan."]
    if invoice_col is None:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), ["Kolom Nomor Invoice di file Invoice tidak ditemukan."]

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__INVOICE__"] = work[invoice_col].apply(normalize_invoice_number)

    work = work[
        work["__PORT__"].notna()
        & work["__AMOUNT__"].notna()
        & work["__INVOICE__"].notna()
    ].copy()

    if date_col is not None:
        parsed = parse_datetime_series(work[date_col])
        work = work[parsed.notna()].copy()
        work = work[between_date_mask(work[date_col], start_date, end_date)].copy()
    else:
        warnings.append("Kolom tanggal di Invoice tidak ditemukan. Data Invoice dipakai seluruhnya untuk pembanding nomor invoice.")

    if work.empty:
        warnings.append(
            f"Tidak ada data Invoice per nomor invoice untuk rentang "
            f"{start_date.strftime('%d-%m-%Y')} s.d. {end_date.strftime('%d-%m-%Y')}."
        )
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), warnings

    grouped = (
        work.groupby(["__PORT__", "__INVOICE__"], as_index=False)["__AMOUNT__"]
        .sum()
        .rename(
            columns={
                "__PORT__": "PELABUHAN",
                "__INVOICE__": "NOMOR_INVOICE",
                "__AMOUNT__": "NOMINAL_INVOICE",
            }
        )
    )

    return grouped, warnings


def aggregate_naik_turun_golongan(
    invoice_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> AggregateResult:
    invoice_grouped, invoice_warnings = aggregate_invoice_by_invoice_number(invoice_df, start_date, end_date)
    summary_grouped, summary_warnings = aggregate_summary_by_invoice_number(summary_df, start_date, end_date)

    warnings = invoice_warnings + summary_warnings

    merged = pd.merge(
        invoice_grouped,
        summary_grouped,
        on=["PELABUHAN", "NOMOR_INVOICE"],
        how="outer",
    )

    if merged.empty:
        return AggregateResult(base_result(), pd.DataFrame(), warnings or ["Tidak ada data Naik Turun Golongan yang bisa dibandingkan."])

    merged["ADA_DI_INVOICE"] = merged["NOMINAL_INVOICE"].notna()
    merged["ADA_DI_TIKET_SUMMARY"] = merged["NOMINAL_TIKET_SUMMARY"].notna()

    merged["NOMINAL_INVOICE"] = merged["NOMINAL_INVOICE"].fillna(0.0)
    merged["NOMINAL_TIKET_SUMMARY"] = merged["NOMINAL_TIKET_SUMMARY"].fillna(0.0)
    merged["SELISIH"] = merged["NOMINAL_INVOICE"] - merged["NOMINAL_TIKET_SUMMARY"]

    merged["STATUS"] = np.select(
        [
            merged["ADA_DI_INVOICE"] & merged["ADA_DI_TIKET_SUMMARY"],
            merged["ADA_DI_INVOICE"] & ~merged["ADA_DI_TIKET_SUMMARY"],
            ~merged["ADA_DI_INVOICE"] & merged["ADA_DI_TIKET_SUMMARY"],
        ],
        [
            "MATCHED",
            "HANYA_DI_INVOICE",
            "HANYA_DI_TIKET_SUMMARY",
        ],
        default="UNKNOWN",
    )

    grouped = merged.groupby("PELABUHAN")["SELISIH"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = merged[
        [
            "PELABUHAN",
            "NOMOR_INVOICE",
            "NOMINAL_INVOICE",
            "NOMINAL_TIKET_SUMMARY",
            "SELISIH",
            "STATUS",
        ]
    ].sort_values(["PELABUHAN", "NOMOR_INVOICE"]).reset_index(drop=True)

    return AggregateResult(result, detail, warnings)


def extract_ticket_sold_structured(df: pd.DataFrame) -> AggregateResult:
    warnings: list[str] = []

    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Terjual kosong."])

    port_col = detect_column(df.columns.tolist(), PORT_CANDIDATES)
    amount_col = detect_column(df.columns.tolist(), AMOUNT_CANDIDATES_TICKET)

    if port_col is None or amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Mode kolom terstruktur Tiket Terjual tidak terdeteksi."])

    work = df.copy()
    work["__ROW_NO__"] = np.arange(len(work))
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
        selected = work[subtotal_mask].sort_values(["__PORT__", "__ROW_NO__"]).groupby("__PORT__", as_index=False).tail(1)
    else:
        selected = work.sort_values(["__PORT__", "__ROW_NO__"]).groupby("__PORT__", as_index=False).tail(1)

    grouped = selected.groupby("__PORT__")["__AMOUNT__"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = selected[[port_col, amount_col, "__PORT__", "__AMOUNT__", "__SOURCE__"]].rename(
        columns={
            port_col: "SUMBER_PORT",
            amount_col: "NOMINAL_SUMBER",
            "__PORT__": "PELABUHAN",
            "__AMOUNT__": "NOMINAL",
            "__SOURCE__": "SUMBER_FILE_SHEET",
        }
    )

    return AggregateResult(result, detail, warnings)


def extract_ticket_sold_report_style(df: pd.DataFrame) -> AggregateResult:
    warnings: list[str] = []

    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Terjual kosong."])

    candidates: list[dict[str, Any]] = []
    current_port: str | None = None

    for idx, row in df.iterrows():
        row_values = row.tolist()
        row_text = " | ".join("" if pd.isna(v) else str(v) for v in row_values)
        normalized_row = normalize_text(row_text)

        explicit_port = canonical_port(normalized_row)
        if explicit_port:
            current_port = explicit_port

        target_port = explicit_port or current_port
        if target_port is None:
            continue

        amount = parse_last_number_from_row(row_values)
        if amount is None:
            continue

        priority = 0
        if "SUB TOTAL" in normalized_row or "SUBTOTAL" in normalized_row:
            priority = 3
        elif "TOTAL JUMLAH" in normalized_row:
            priority = 2
        elif explicit_port is not None:
            priority = 1

        if priority == 0:
            continue

        candidates.append(
            {
                "ROW_INDEX": idx,
                "PELABUHAN": target_port,
                "NOMINAL": float(amount),
                "PRIORITY": priority,
                "BARIS_SUMBER": row_text,
                "SUMBER_FILE_SHEET": row.get("__SOURCE__", ""),
            }
        )

    if not candidates:
        return AggregateResult(base_result(), pd.DataFrame(), ["Parser subtotal/total Tiket Terjual tidak menemukan baris yang cocok."])

    candidate_df = pd.DataFrame(candidates)
    candidate_df = candidate_df.sort_values(["PELABUHAN", "PRIORITY", "ROW_INDEX"], ascending=[True, False, False])

    selected_rows: list[pd.Series] = []
    for port in DEFAULT_PORTS:
        subset = candidate_df[candidate_df["PELABUHAN"] == port]
        if not subset.empty:
            selected_rows.append(subset.iloc[0])

    if not selected_rows:
        return AggregateResult(base_result(), pd.DataFrame(), ["Tidak ada total Tiket Terjual yang cocok per pelabuhan."])

    selected = pd.DataFrame(selected_rows)
    grouped = selected.groupby("PELABUHAN")["NOMINAL"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    return AggregateResult(result, selected.reset_index(drop=True), warnings)


def extract_ticket_sold_totals(df: pd.DataFrame) -> AggregateResult:
    structured = extract_ticket_sold_structured(df)
    if structured.series.sum() != 0:
        return structured

    fallback = extract_ticket_sold_report_style(df)
    if fallback.series.sum() != 0:
        return fallback

    warnings = structured.warnings + fallback.warnings
    return AggregateResult(base_result(), pd.DataFrame(), warnings)


def build_reconciliation(
    ticket_sold_df: pd.DataFrame,
    ticket_summary_df: pd.DataFrame,
    invoice_df: pd.DataFrame,
    addition_date: date,
    deduction_date: date,
    ntg_start_date: date,
    ntg_end_date: date,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    warnings: list[str] = []

    ticket_sold = extract_ticket_sold_totals(ticket_sold_df)
    addition = aggregate_summary_window(ticket_summary_df, addition_date)
    deduction = aggregate_summary_window(ticket_summary_df, deduction_date)
    naik_turun = aggregate_naik_turun_golongan(invoice_df, ticket_summary_df, ntg_start_date, ntg_end_date)

    warnings.extend(ticket_sold.warnings)
    warnings.extend(addition.warnings)
    warnings.extend(deduction.warnings)
    warnings.extend(naik_turun.warnings)

    result = pd.DataFrame({"Pelabuhan (ASAL)": DEFAULT_PORTS})
    result["Nominal Tiket Terjual"] = result["Pelabuhan (ASAL)"].map(ticket_sold.series).fillna(0.0)
    result["Nominal Penambahan"] = result["Pelabuhan (ASAL)"].map(addition.series).fillna(0.0)
    result["Nominal Pengurangan"] = result["Pelabuhan (ASAL)"].map(deduction.series).fillna(0.0)
    result["Nominal Naik Turun Golongan"] = result["Pelabuhan (ASAL)"].map(naik_turun.series).fillna(0.0)
    result["Nominal Pinbuk"] = (
        result["Nominal Tiket Terjual"]
        + result["Nominal Penambahan"]
        - result["Nominal Pengurangan"]
        + result["Nominal Naik Turun Golongan"]
    )

    details = {
        "tiket_terjual": ticket_sold.detail,
        "penambahan": addition.detail,
        "pengurangan": deduction.detail,
        "naik_turun_golongan": naik_turun.detail,
    }

    unique_warnings = list(dict.fromkeys(warnings))
    return result, details, unique_warnings


def format_currency_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else "")
    return result


def to_excel_bytes(reconciliation_df: pd.DataFrame, detail_tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        reconciliation_df.to_excel(writer, index=False, sheet_name="Rekonsiliasi")
        for name, detail_df in detail_tables.items():
            detail_df.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output.getvalue()


def file_section(label: str, key_prefix: str) -> tuple[dict[str, pd.DataFrame] | None, list[str]]:
    uploaded_files = st.file_uploader(
        label,
        type=["xlsx", "xls", "xlsm", "csv"],
        accept_multiple_files=True,
        key=f"{key_prefix}_uploader",
    )

    if not uploaded_files:
        return None, []

    sheet_map, errors = load_multiple_files(uploaded_files)
    for error in errors:
        st.error(f"Gagal membaca {label}: {error}")

    if not sheet_map:
        return None, []

    options = list(sheet_map.keys())
    selected = st.multiselect(
        f"Pilih file/sheet untuk {label}",
        options=options,
        default=options,
        key=f"{key_prefix}_selected",
    )
    return sheet_map, selected


def render_preview(name: str, sheet_map: dict[str, pd.DataFrame] | None, selected: list[str]) -> pd.DataFrame:
    if not sheet_map or not selected:
        return pd.DataFrame()

    combined = combine_selected_sheets(sheet_map, selected)
    file_count = combined["__FILE__"].nunique() if "__FILE__" in combined.columns else 0

    with st.expander(f"Preview {name}", expanded=False):
        st.caption(
            f"{file_count} file | {len(selected)} file/sheet dipilih | "
            f"{combined.shape[0]} baris | {combined.shape[1]} kolom"
        )
        st.dataframe(combined.head(30), use_container_width=True, height=280)

    return combined


st.set_page_config(page_title="Rekonsiliasi Sales Channel", layout="wide")
st.title("Rekonsiliasi Sales Channel")
st.caption(
    "Multiple upload aktif. "
    "Penambahan/Pengurangan memakai CETAK BOARDING PASS pada jam 00:00:00 - 07:59:59. "
    "Naik Turun Golongan = selisih nominal per nomor invoice antara Invoice dan Tiket Summary, lalu dijumlah per pelabuhan."
)

with st.sidebar:
    st.subheader("Parameter")

    addition_date = st.date_input("Tanggal Penambahan", value=date.today(), key="addition_date")
    deduction_date = st.date_input("Tanggal Pengurangan", value=date.today(), key="deduction_date")
    ntg_start_date = st.date_input("Tanggal Naik Turun Golongan - Mulai", value=date.today(), key="ntg_start_date")
    ntg_end_date = st.date_input("Tanggal Naik Turun Golongan - Selesai", value=date.today(), key="ntg_end_date")

    st.markdown("---")
    st.write("Window Penambahan/Pengurangan:")
    st.write(f"{WINDOW_START} - {WINDOW_END}")
    st.markdown("---")
    st.write("Pelabuhan default:")
    st.write(", ".join(DEFAULT_PORTS))

col1, col2, col3 = st.columns(3)

with col1:
    ticket_sold_map, ticket_sold_selected = file_section("Uploader Tiket Terjual", "ticket_sold")

with col2:
    ticket_summary_map, ticket_summary_selected = file_section("Uploader Tiket Summary", "ticket_summary")

with col3:
    invoice_map, invoice_selected = file_section("Uploader Invoice", "invoice")

ticket_sold_df = render_preview("Tiket Terjual", ticket_sold_map, ticket_sold_selected)
ticket_summary_df = render_preview("Tiket Summary", ticket_summary_map, ticket_summary_selected)
invoice_df = render_preview("Invoice", invoice_map, invoice_selected)

ready = all(
    [
        ticket_sold_map is not None,
        ticket_summary_map is not None,
        invoice_map is not None,
        len(ticket_sold_selected) > 0,
        len(ticket_summary_selected) > 0,
        len(invoice_selected) > 0,
    ]
)

if not ready:
    st.info("Lengkapi 3 uploader dan pilih minimal 1 file/sheet pada masing-masing uploader.")
else:
    reconciliation_df, detail_tables, warnings_list = build_reconciliation(
        ticket_sold_df=ticket_sold_df,
        ticket_summary_df=ticket_summary_df,
        invoice_df=invoice_df,
        addition_date=addition_date,
        deduction_date=deduction_date,
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
    st.dataframe(
        format_currency_columns(reconciliation_df, numeric_columns),
        use_container_width=True,
        height=360,
    )

    total_row = reconciliation_df[numeric_columns].sum().to_frame().T
    total_row.insert(0, "Pelabuhan (ASAL)", "TOTAL")

    st.subheader("Grand Total")
    st.dataframe(
        format_currency_columns(total_row, numeric_columns),
        use_container_width=True,
        hide_index=True,
    )

    if warnings_list:
        with st.expander("Catatan parser", expanded=False):
            for warning in warnings_list:
                st.warning(warning)

    excel_bytes = to_excel_bytes(reconciliation_df, detail_tables)
    st.download_button(
        "Download Hasil Rekonsiliasi (Excel)",
        data=excel_bytes,
        file_name="rekonsiliasi_sales_channel.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tabs = st.tabs(
        [
            "Detail Tiket Terjual",
            "Detail Penambahan",
            "Detail Pengurangan",
            "Detail Naik Turun Golongan",
        ]
    )
    tab_keys = ["tiket_terjual", "penambahan", "pengurangan", "naik_turun_golongan"]

    for tab, key in zip(tabs, tab_keys):
        with tab:
            detail_df = detail_tables[key]
            if detail_df.empty:
                st.info("Tidak ada data detail.")
            else:
                amount_like_columns = [
                    col for col in detail_df.columns
                    if "NOMINAL" in col.upper() or "SELISIH" in col.upper()
                ]
                st.dataframe(
                    format_currency_columns(detail_df, amount_like_columns),
                    use_container_width=True,
                    height=420,
                )

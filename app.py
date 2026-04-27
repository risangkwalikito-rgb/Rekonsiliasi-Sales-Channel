# file: app.py

import io
import re
from dataclasses import dataclass
from datetime import date, datetime
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

SUMMARY_BP_DATE_COL = "CETAK BOARDING PASS"
SUMMARY_PAYMENT_DATE_COL = "TANGGAL PEMBAYARAN"
SUMMARY_AMOUNT_COL = "TARIF"

INVOICE_DATE_COL = "TANGGAL INVOICE"
INVOICE_AMOUNT_COL = "HARGA"

PORT_COL_CANDIDATES = ["ASAL", "PELABUHAN", "CABANG", "PELABUHAN (ASAL)"]
INVOICE_NO_CANDIDATES = [
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

TICKET_SOLD_PORT_CANDIDATES = ["CABANG", "PELABUHAN", "ASAL", "PELABUHAN (ASAL)"]
TICKET_SOLD_AMOUNT_CANDIDATES = [
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

NUMERIC_COLUMNS = [
    "Nominal Tiket Terjual",
    "Nominal Penambahan",
    "Nominal Pengurangan",
    "Nominal Naik Turun Golongan",
    "Nominal Pinbuk",
]


@dataclass
class AggregateResult:
    series: pd.Series
    detail: pd.DataFrame
    warnings: list[str]


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def normalize_header(value: Any) -> str:
    text = normalize_text(value).replace("_", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def canonical_port(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    for canonical, aliases in PORT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def normalize_invoice_number(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)

    if text in {"", "NAN", "NONE", "-"}:
        return None

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def clean_money_text(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("Rp", "").replace("rp", "")
    text = text.replace(" ", "")
    return re.sub(r"[^\d,.\-]", "", text)


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
            text = text.replace(".", "").replace(",", ".")
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
    numbers = [parse_number(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if numbers:
        return numbers[-1]

    row_text = " ".join(str(v) for v in values if v is not None and not pd.isna(v))
    matches = re.findall(r"-?\d[\d.,]*", row_text)
    if not matches:
        return None
    return parse_number(matches[-1])


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_header(col) for col in result.columns]
    result = result.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return result.reset_index(drop=True)


def require_column(df: pd.DataFrame, column_name: str) -> str | None:
    target = normalize_header(column_name)
    for col in df.columns:
        if normalize_header(col) == target:
            return col
    return None


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_columns = [normalize_header(col) for col in columns]
    normalized_candidates = [normalize_header(c) for c in candidates]

    for candidate in normalized_candidates:
        for index, column in enumerate(normalized_columns):
            if column == candidate:
                return columns[index]

    for candidate in normalized_candidates:
        for index, column in enumerate(normalized_columns):
            if candidate in column:
                return columns[index]

    return None


def detect_invoice_no_column(df: pd.DataFrame) -> str | None:
    detected = detect_column(list(df.columns), INVOICE_NO_CANDIDATES)
    if detected:
        return detected

    for col in df.columns:
        norm = normalize_header(col)
        if "INVOICE" in norm and "TANGGAL" not in norm and "DATE" not in norm:
            return col
        if "INV" in norm and "TANGGAL" not in norm and "DATE" not in norm:
            return col
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
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ]
    as_text = series.astype(str)

    for fmt in formats:
        parsed = pd.to_datetime(as_text, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed

    return pd.to_datetime(as_text, errors="coerce")


def between_date_mask(series: pd.Series, start_date: date, end_date: date) -> pd.Series:
    parsed = parse_datetime_series(series)
    if parsed.notna().sum() == 0:
        return pd.Series(False, index=series.index)
    return (parsed.dt.date >= start_date) & (parsed.dt.date <= end_date)


def exact_date_time_window_mask(
    series: pd.Series,
    target_date: date,
    start_time: str = WINDOW_START,
    end_time: str = WINDOW_END,
) -> pd.Series:
    parsed = parse_datetime_series(series)
    if parsed.notna().sum() == 0:
        return pd.Series(False, index=series.index)

    start = datetime.strptime(start_time, "%H:%M:%S").time()
    end = datetime.strptime(end_time, "%H:%M:%S").time()
    return (parsed.dt.date == target_date) & (parsed.dt.time >= start) & (parsed.dt.time <= end)


def base_result() -> pd.Series:
    return pd.Series(0.0, index=DEFAULT_PORTS, dtype=float)


def append_total_row(df: pd.DataFrame, label_col: str, numeric_columns: list[str]) -> pd.DataFrame:
    total_row = {label_col: "TOTAL"}
    for col in numeric_columns:
        total_row[col] = df[col].sum() if col in df.columns else 0.0
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def read_uploaded_file_first_sheet(uploaded_file: Any) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    buffer = io.BytesIO(content)

    if file_name.endswith((".xlsx", ".xls", ".xlsm")):
        excel_file = pd.ExcelFile(buffer)
        first_sheet = excel_file.sheet_names[0]
        df = pd.read_excel(excel_file, sheet_name=first_sheet, dtype=object)
        result = prepare_dataframe(df)
        result["__FILE__"] = uploaded_file.name
        result["__SHEET__"] = first_sheet
        result["__SOURCE__"] = f"{uploaded_file.name} :: {first_sheet}"
        return result

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(buffer, dtype=object)
        except Exception:
            buffer.seek(0)
            df = pd.read_csv(buffer, dtype=object, sep=";")
        result = prepare_dataframe(df)
        result["__FILE__"] = uploaded_file.name
        result["__SHEET__"] = "CSV"
        result["__SOURCE__"] = f"{uploaded_file.name} :: CSV"
        return result

    raise ValueError("Format file tidak didukung. Gunakan Excel atau CSV.")


def load_multiple_files_first_sheet(uploaded_files: list[Any]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            frames.append(read_uploaded_file_first_sheet(uploaded_file))
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    if not frames:
        return pd.DataFrame(), errors

    return pd.concat(frames, ignore_index=True, sort=False), errors


def aggregate_summary_window(df: pd.DataFrame, target_date: date) -> AggregateResult:
    warnings: list[str] = []

    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Summary kosong."])

    date_col = require_column(df, SUMMARY_BP_DATE_COL)
    amount_col = require_column(df, SUMMARY_AMOUNT_COL)
    port_col = detect_column(list(df.columns), PORT_COL_CANDIDATES)

    missing = []
    if date_col is None:
        missing.append(f"`{SUMMARY_BP_DATE_COL}`")
    if amount_col is None:
        missing.append(f"`{SUMMARY_AMOUNT_COL}`")
    if port_col is None:
        missing.append("kolom ASAL/Pelabuhan")

    if missing:
        return AggregateResult(
            base_result(),
            pd.DataFrame(),
            [f"Tiket Summary untuk Penambahan/Pengurangan tidak memiliki: {', '.join(missing)}."],
        )

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__DATETIME__"] = parse_datetime_series(work[date_col])

    work = work[
        work["__PORT__"].notna()
        & work["__AMOUNT__"].notna()
        & work["__DATETIME__"].notna()
    ].copy()

    work = work[exact_date_time_window_mask(work[date_col], target_date)].copy()

    grouped = work.groupby("__PORT__")["__AMOUNT__"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = work[[date_col, port_col, amount_col, "__PORT__", "__AMOUNT__", "__SOURCE__"]].rename(
        columns={
            date_col: "CETAK_BOARDING_PASS",
            port_col: "ASAL_SUMBER",
            amount_col: "TARIF_SUMBER",
            "__PORT__": "PELABUHAN",
            "__AMOUNT__": "NOMINAL",
            "__SOURCE__": "SUMBER_FILE_SHEET",
        }
    )

    if detail.empty:
        warnings.append(
            f"Tidak ada data Tiket Summary pada {target_date.strftime('%Y/%m/%d')} "
            f"jam {WINDOW_START}-{WINDOW_END} berdasarkan kolom `{SUMMARY_BP_DATE_COL}`."
        )

    return AggregateResult(result, detail, warnings)


def aggregate_summary_by_invoice(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if df.empty:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), ["File Tiket Summary kosong."]

    date_col = require_column(df, SUMMARY_PAYMENT_DATE_COL)
    amount_col = require_column(df, SUMMARY_AMOUNT_COL)
    port_col = detect_column(list(df.columns), PORT_COL_CANDIDATES)
    invoice_col = detect_invoice_no_column(df)

    missing = []
    if date_col is None:
        missing.append(f"`{SUMMARY_PAYMENT_DATE_COL}`")
    if amount_col is None:
        missing.append(f"`{SUMMARY_AMOUNT_COL}`")
    if port_col is None:
        missing.append("kolom ASAL/Pelabuhan")
    if invoice_col is None:
        missing.append("kolom Nomor Invoice")

    if missing:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_TIKET_SUMMARY"]), [
            f"Tiket Summary untuk Naik Turun Golongan tidak memiliki: {', '.join(missing)}."
        ]

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__INVOICE__"] = work[invoice_col].apply(normalize_invoice_number)
    work["__DATE__"] = parse_datetime_series(work[date_col])

    work = work[
        work["__PORT__"].notna()
        & work["__AMOUNT__"].notna()
        & work["__INVOICE__"].notna()
        & work["__DATE__"].notna()
    ].copy()

    work = work[between_date_mask(work[date_col], start_date, end_date)].copy()

    if work.empty:
        warnings.append(
            "Tidak ada data Tiket Summary yang sesuai parameter tanggal "
            f"berdasarkan kolom `{SUMMARY_PAYMENT_DATE_COL}` untuk perbandingan nomor invoice."
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


def aggregate_invoice_by_invoice(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if df.empty:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), ["File Invoice kosong."]

    date_col = require_column(df, INVOICE_DATE_COL)
    amount_col = require_column(df, INVOICE_AMOUNT_COL)
    port_col = detect_column(list(df.columns), PORT_COL_CANDIDATES)
    invoice_col = detect_invoice_no_column(df)

    missing = []
    if date_col is None:
        missing.append(f"`{INVOICE_DATE_COL}`")
    if amount_col is None:
        missing.append(f"`{INVOICE_AMOUNT_COL}`")
    if port_col is None:
        missing.append("kolom ASAL/Pelabuhan")
    if invoice_col is None:
        missing.append("kolom Nomor Invoice")

    if missing:
        return pd.DataFrame(columns=["PELABUHAN", "NOMOR_INVOICE", "NOMINAL_INVOICE"]), [
            f"Invoice tidak memiliki: {', '.join(missing)}."
        ]

    work = df.copy()
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__INVOICE__"] = work[invoice_col].apply(normalize_invoice_number)
    work["__DATE__"] = parse_datetime_series(work[date_col])

    work = work[
        work["__PORT__"].notna()
        & work["__AMOUNT__"].notna()
        & work["__INVOICE__"].notna()
        & work["__DATE__"].notna()
    ].copy()

    work = work[between_date_mask(work[date_col], start_date, end_date)].copy()

    if work.empty:
        warnings.append(
            "Tidak ada data Invoice yang sesuai parameter tanggal "
            f"berdasarkan kolom `{INVOICE_DATE_COL}` untuk perbandingan nomor invoice."
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
    invoice_grouped, invoice_warnings = aggregate_invoice_by_invoice(invoice_df, start_date, end_date)
    summary_grouped, summary_warnings = aggregate_summary_by_invoice(summary_df, start_date, end_date)
    warnings = invoice_warnings + summary_warnings

    merged = pd.merge(
        invoice_grouped,
        summary_grouped,
        on=["PELABUHAN", "NOMOR_INVOICE"],
        how="outer",
    )

    if merged.empty:
        return AggregateResult(
            base_result(),
            pd.DataFrame(),
            warnings or ["Tidak ada data Naik Turun Golongan yang bisa dibandingkan."],
        )

    merged["NOMINAL_INVOICE"] = merged["NOMINAL_INVOICE"].fillna(0.0)
    merged["NOMINAL_TIKET_SUMMARY"] = merged["NOMINAL_TIKET_SUMMARY"].fillna(0.0)
    merged["SELISIH"] = merged["NOMINAL_INVOICE"] - merged["NOMINAL_TIKET_SUMMARY"]

    grouped = merged.groupby("PELABUHAN")["SELISIH"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = merged.sort_values(["PELABUHAN", "NOMOR_INVOICE"]).reset_index(drop=True)
    return AggregateResult(result, detail, warnings)


def extract_ticket_sold_structured(df: pd.DataFrame) -> AggregateResult:
    warnings: list[str] = []

    if df.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["File Tiket Terjual kosong."])

    port_col = detect_column(list(df.columns), TICKET_SOLD_PORT_CANDIDATES)
    amount_col = detect_column(list(df.columns), TICKET_SOLD_AMOUNT_CANDIDATES)

    if port_col is None or amount_col is None:
        return AggregateResult(base_result(), pd.DataFrame(), ["Mode kolom terstruktur Tiket Terjual tidak terdeteksi."])

    work = df.copy()
    work["__ROW_NO__"] = np.arange(len(work))
    work["__PORT__"] = work[port_col].apply(canonical_port)
    work["__AMOUNT__"] = work[amount_col].apply(parse_number)
    work["__ROW_TEXT__"] = work.astype(str).agg(" | ".join, axis=1).apply(normalize_text)

    work = work[work["__PORT__"].notna() & work["__AMOUNT__"].notna()].copy()
    if work.empty:
        return AggregateResult(base_result(), pd.DataFrame(), ["Tidak ada data Tiket Terjual yang cocok."])

    subtotal_mask = (
        work["__ROW_TEXT__"].str.contains("SUB TOTAL", regex=False, na=False)
        | work["__ROW_TEXT__"].str.contains("SUBTOTAL", regex=False, na=False)
        | work["__ROW_TEXT__"].str.contains("TOTAL JUMLAH", regex=False, na=False)
    )

    selector = work[subtotal_mask].copy() if subtotal_mask.any() else work.copy()

    selected = (
        selector.sort_values(["__SOURCE__", "__PORT__", "__ROW_NO__"])
        .groupby(["__SOURCE__", "__PORT__"], as_index=False)
        .tail(1)
    )

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
    current_port_by_source: dict[str, str | None] = {}

    for idx, row in df.iterrows():
        source = row.get("__SOURCE__", "")
        current_port = current_port_by_source.get(source)

        row_values = row.tolist()
        row_text = " | ".join("" if pd.isna(v) else str(v) for v in row_values)
        normalized_row = normalize_text(row_text)

        explicit_port = canonical_port(normalized_row)
        if explicit_port:
            current_port = explicit_port
            current_port_by_source[source] = explicit_port

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
                "SOURCE": source,
                "PELABUHAN": target_port,
                "NOMINAL": float(amount),
                "PRIORITY": priority,
                "BARIS_SUMBER": row_text,
                "SUMBER_FILE_SHEET": source,
            }
        )

    if not candidates:
        return AggregateResult(
            base_result(),
            pd.DataFrame(),
            ["Parser subtotal/total Tiket Terjual tidak menemukan data yang cocok."],
        )

    candidate_df = pd.DataFrame(candidates).sort_values(
        ["SOURCE", "PELABUHAN", "PRIORITY", "ROW_INDEX"],
        ascending=[True, True, False, False],
    )

    selected = candidate_df.groupby(["SOURCE", "PELABUHAN"], as_index=False).head(1).reset_index(drop=True)

    grouped = selected.groupby("PELABUHAN")["NOMINAL"].sum()
    result = base_result().add(grouped, fill_value=0.0)

    detail = selected[
        ["SUMBER_FILE_SHEET", "PELABUHAN", "NOMINAL", "PRIORITY", "ROW_INDEX", "BARIS_SUMBER"]
    ]

    return AggregateResult(result, detail, warnings)


def extract_ticket_sold_totals(df: pd.DataFrame) -> AggregateResult:
    structured = extract_ticket_sold_structured(df)
    if structured.series.sum() != 0:
        return structured

    fallback = extract_ticket_sold_report_style(df)
    if fallback.series.sum() != 0:
        return fallback

    return AggregateResult(base_result(), pd.DataFrame(), structured.warnings + fallback.warnings)


def normalize_date_range_input(value: Any) -> tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0], value[0]
    return value, value


def build_reconciliation(
    ticket_sold_df: pd.DataFrame,
    ticket_summary_df: pd.DataFrame,
    invoice_df: pd.DataFrame,
    addition_date: date,
    deduction_date: date,
    ntg_start_date: date,
    ntg_end_date: date,
) -> pd.DataFrame:
    ticket_sold = extract_ticket_sold_totals(ticket_sold_df)
    addition = aggregate_summary_window(ticket_summary_df, addition_date)
    deduction = aggregate_summary_window(ticket_summary_df, deduction_date)
    naik_turun = aggregate_naik_turun_golongan(invoice_df, ticket_summary_df, ntg_start_date, ntg_end_date)

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

    return append_total_row(
        result,
        label_col="Pelabuhan (ASAL)",
        numeric_columns=NUMERIC_COLUMNS,
    )


def format_currency_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(
                lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else ""
            )
    return result


def empty_reconciliation_table() -> pd.DataFrame:
    empty_df = pd.DataFrame(
        {
            "Pelabuhan (ASAL)": DEFAULT_PORTS,
            "Nominal Tiket Terjual": [0.0] * len(DEFAULT_PORTS),
            "Nominal Penambahan": [0.0] * len(DEFAULT_PORTS),
            "Nominal Pengurangan": [0.0] * len(DEFAULT_PORTS),
            "Nominal Naik Turun Golongan": [0.0] * len(DEFAULT_PORTS),
            "Nominal Pinbuk": [0.0] * len(DEFAULT_PORTS),
        }
    )
    return append_total_row(
        empty_df,
        label_col="Pelabuhan (ASAL)",
        numeric_columns=NUMERIC_COLUMNS,
    )


def uploader_first_sheet(label: str, key_prefix: str) -> pd.DataFrame:
    uploaded_files = st.sidebar.file_uploader(
        label,
        type=["xlsx", "xls", "xlsm", "csv"],
        accept_multiple_files=True,
        key=f"{key_prefix}_uploader",
    )

    if not uploaded_files:
        return pd.DataFrame()

    combined_df, _errors = load_multiple_files_first_sheet(uploaded_files)
    return combined_df


st.set_page_config(page_title="Rekonsiliasi Sales Channel", layout="wide")
st.title("Rekonsiliasi Sales Channel")

today = date.today()

with st.sidebar:
    st.subheader("Parameter")

    penambahan_pengurangan_range = st.date_input(
        "Rentang Penambahan / Pengurangan",
        value=(today, today),
        key="penambahan_pengurangan_range",
    )
    ntg_range = st.date_input(
        "Rentang Naik Turun Golongan",
        value=(today, today),
        key="ntg_range",
    )

    st.divider()
    st.subheader("Uploader")
    st.caption("Excel otomatis memakai sheet 1.")

addition_date, deduction_date = normalize_date_range_input(penambahan_pengurangan_range)
ntg_start_date, ntg_end_date = normalize_date_range_input(ntg_range)

ticket_sold_df = uploader_first_sheet("Tiket Terjual", "ticket_sold")
ticket_summary_df = uploader_first_sheet("Tiket Summary", "ticket_summary")
invoice_df = uploader_first_sheet("Invoice", "invoice")

ready = not ticket_sold_df.empty and not ticket_summary_df.empty and not invoice_df.empty

if ready:
    reconciliation_df = build_reconciliation(
        ticket_sold_df=ticket_sold_df,
        ticket_summary_df=ticket_summary_df,
        invoice_df=invoice_df,
        addition_date=addition_date,
        deduction_date=deduction_date,
        ntg_start_date=ntg_start_date,
        ntg_end_date=ntg_end_date,
    )
else:
    reconciliation_df = empty_reconciliation_table()

st.dataframe(
    format_currency_columns(reconciliation_df, NUMERIC_COLUMNS),
    use_container_width=True,
    height=420,
)

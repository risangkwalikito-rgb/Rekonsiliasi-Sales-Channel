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

    return result, details, list(dict.fromkeys(warnings))


def format_currency_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(
                lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else ""
            )
    return result


def to_excel_bytes(reconciliation_df: pd.DataFrame, detail_tables: dict[str, pd.DataFrame]) -> bytes:
    export_df = append_total_row(
        reconciliation_df,
        label_col="Pelabuhan (ASAL)",
        numeric_columns=NUMERIC_COLUMNS,
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Rekonsiliasi")
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
        st.write("Kolom:", list(combined.columns))

    return combined


def render_validation(summary_df: pd.DataFrame, invoice_df: pd.DataFrame) -> None:
    with st.expander("Validasi kolom wajib", expanded=False):
        summary_required = [SUMMARY_BP_DATE_COL, SUMMARY_PAYMENT_DATE_COL, SUMMARY_AMOUNT_COL]
        invoice_required = [INVOICE_DATE_COL, INVOICE_AMOUNT_COL]

        summary_found = [col for col in summary_required if require_column(summary_df, col)]
        summary_missing = [col for col in summary_required if not require_column(summary_df, col)]
        invoice_found = [col for col in invoice_required if require_column(invoice_df, col)]
        invoice_missing = [col for col in invoice_required if not require_column(invoice_df, col)]

        summary_invoice_col = detect_invoice_no_column(summary_df)
        invoice_invoice_col = detect_invoice_no_column(invoice_df)

        st.write("Tiket Summary ditemukan:", summary_found or "-")
        st.write("Tiket Summary belum ada:", summary_missing or "-")
        st.write("Tiket Summary Nomor Invoice:", summary_invoice_col or "BELUM DITEMUKAN")
        st.write("Invoice ditemukan:", invoice_found or "-")
        st.write("Invoice belum ada:", invoice_missing or "-")
        st.write("Invoice Nomor Invoice:", invoice_invoice_col or "BELUM DITEMUKAN")


st.set_page_config(page_title="Rekonsiliasi Sales Channel", layout="wide")
st.title("Rekonsiliasi Sales Channel")
st.caption(
    "Penambahan/Pengurangan = Tiket Summary[`CETAK BOARDING PASS`, `Tarif`] pada jam 00:00:00-07:59:59 | "
    "Naik Turun Golongan = compare per Nomor Invoice antara Invoice[`Tanggal Invoice`, `Harga`] "
    "vs Tiket Summary[`Tanggal Pembayaran`, `Tarif`] | "
    "Baris TOTAL ada di tabel yang sama."
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

if not ticket_summary_df.empty and not invoice_df.empty:
    render_validation(ticket_summary_df, invoice_df)

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

    display_df = append_total_row(
        reconciliation_df,
        label_col="Pelabuhan (ASAL)",
        numeric_columns=NUMERIC_COLUMNS,
    )

    st.subheader("Tabel Rekonsiliasi Sales Channel")
    st.dataframe(
        format_currency_columns(display_df, NUMERIC_COLUMNS),
        use_container_width=True,
        height=440,
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
                amount_cols = [
                    col
                    for col in detail_df.columns
                    if "NOMINAL" in col.upper() or "SELISIH" in col.upper() or "TARIF" in col.upper()
                ]
                st.dataframe(
                    format_currency_columns(detail_df, amount_cols),
                    use_container_width=True,
                    height=420,
                )

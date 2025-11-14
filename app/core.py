"""Hand-written reconciliation helpers with just enough structure to stay tidy."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from io import TextIOBase, TextIOWrapper
from typing import BinaryIO, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from dateutil import parser as date_parser

# === Data models ===


@dataclass(frozen=True, slots=True)
class Invoice:
    invoice_id: str
    customer_name: str
    invoice_date: date
    due_date: Optional[date]
    currency: str
    amount: Decimal


@dataclass(frozen=True, slots=True)
class Payment:
    payment_id: str
    payer_name: str
    payment_date: date
    currency: str
    amount: Decimal
    reference: str


@dataclass(frozen=True, slots=True)
class MatchRecord:
    invoice: Invoice
    payment: Payment
    rule: str
    explanation: str


@dataclass(slots=True)
class ReconciliationResult:
    matches: List[MatchRecord]
    unmatched_invoices: List[Invoice]
    unmatched_payments: List[Payment]

    def summary(self) -> dict:
        return {
            "invoices": len(self.matches) + len(self.unmatched_invoices),
            "payments": len(self.matches) + len(self.unmatched_payments),
            "matched_pairs": len(self.matches),
            "unmatched_invoices": len(self.unmatched_invoices),
            "unmatched_payments": len(self.unmatched_payments),
        }


# === Parsing utilities ===


class ParsingError(ValueError):
    """Raised when CSV input cannot be processed."""


INVOICE_COLUMNS = ("invoice_id", "customer_name", "invoice_date", "due_date", "currency", "amount")
PAYMENT_COLUMNS = ("payment_id", "payer_name", "payment_date", "currency", "amount", "reference")


def _parse_date(value: str, *, field_name: str) -> date:
    if not value:
        raise ParsingError(f"Missing value for {field_name}")
    try:
        return date_parser.isoparse(value).date()
    except (ValueError, TypeError):
        raise ParsingError(f"Invalid date '{value}' for {field_name}")


def _parse_amount(value: str, *, field_name: str) -> Decimal:
    if not value:
        raise ParsingError(f"Missing value for {field_name}")
    normalized = value.replace(",", "").strip()
    try:
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        raise ParsingError(f"Invalid amount '{value}' for {field_name}")


def _normalize_currency(value: str) -> str:
    if not value:
        raise ParsingError("Currency is required")
    return value.strip().upper()


def _read_csv(file_obj: TextIOBase, required_columns: Sequence[str]) -> Iterable[dict]:
    reader = csv.DictReader(file_obj)
    missing = [col for col in required_columns if col not in (reader.fieldnames or [])]
    if missing:
        raise ParsingError(f"Missing required columns: {', '.join(missing)}")
    for line_number, row in enumerate(reader, start=2):  # header is line 1
        if not any(row.values()):
            continue
        cleaned = {k: (v or "").strip() for k, v in row.items()}
        cleaned["_line_number"] = line_number
        yield cleaned


def load_invoices(file_obj: TextIOBase) -> List[Invoice]:
    invoices: List[Invoice] = []
    for row in _read_csv(file_obj, INVOICE_COLUMNS):
        line_number = row.pop("_line_number", None)
        try:
            invoices.append(
                Invoice(
                    invoice_id=row["invoice_id"],
                    customer_name=row["customer_name"],
                    invoice_date=_parse_date(row["invoice_date"], field_name="invoice_date"),
                    due_date=_parse_date(row["due_date"], field_name="due_date") if row["due_date"] else None,
                    currency=_normalize_currency(row["currency"]),
                    amount=_parse_amount(row["amount"], field_name="amount"),
                )
            )
        except ParsingError as exc:
            prefix = f"Invoices line {line_number}: " if line_number else "Invoices: "
            raise ParsingError(prefix + str(exc)) from exc
    if not invoices:
        raise ParsingError("Invoices file is empty.")
    return invoices


def load_payments(file_obj: TextIOBase) -> List[Payment]:
    payments: List[Payment] = []
    for row in _read_csv(file_obj, PAYMENT_COLUMNS):
        line_number = row.pop("_line_number", None)
        try:
            payments.append(
                Payment(
                    payment_id=row["payment_id"],
                    payer_name=row["payer_name"],
                    payment_date=_parse_date(row["payment_date"], field_name="payment_date"),
                    currency=_normalize_currency(row["currency"]),
                    amount=_parse_amount(row["amount"], field_name="amount"),
                    reference=row["reference"],
                )
            )
        except ParsingError as exc:
            prefix = f"Payments line {line_number}: " if line_number else "Payments: "
            raise ParsingError(prefix + str(exc)) from exc
    if not payments:
        raise ParsingError("Payments file is empty.")
    return payments


# === Matching engine ===


RuleCandidate = Tuple[Payment, str, str]  # payment, rule name, explanation


def _score_exact(invoice: Invoice, payment: Payment) -> Tuple[int, int]:
    delta = (payment.payment_date - invoice.invoice_date).days
    return (abs(delta), 0 if delta >= 0 else 1)


def _score_window(invoice: Invoice, payment: Payment) -> Tuple[int, int]:
    delta = (payment.payment_date - invoice.invoice_date).days
    return (abs(delta), 0 if delta >= 0 else 1)


def _score_reference(invoice: Invoice, payment: Payment) -> Tuple[int, int]:
    return _score_window(invoice, payment)


class ReconciliationEngine:
    """Tiny rule runner. It's meant to be obvious rather than clever."""

    def __init__(self, *, date_window_days: int = 30, enable_reference_rule: bool = True):
        self.date_window = timedelta(days=date_window_days)
        self.enable_reference_rule = enable_reference_rule

    def reconcile(self, invoices: Sequence[Invoice], payments: Sequence[Payment]) -> ReconciliationResult:
        used_payment_ids: set[str] = set()
        matches: List[MatchRecord] = []
        unmatched_invoices: List[Invoice] = []

        for invoice in sorted(invoices, key=lambda i: i.invoice_date):
            candidate = self._match_invoice(invoice, payments, used_payment_ids)
            if candidate:
                payment, rule, explanation = candidate
                used_payment_ids.add(payment.payment_id)
                matches.append(MatchRecord(invoice=invoice, payment=payment, rule=rule, explanation=explanation))
            else:
                unmatched_invoices.append(invoice)

        unmatched_payments = [p for p in payments if p.payment_id not in used_payment_ids]
        return ReconciliationResult(matches=matches, unmatched_invoices=unmatched_invoices, unmatched_payments=unmatched_payments)

    def _match_invoice(
        self, invoice: Invoice, payments: Sequence[Payment], used_payment_ids: set[str]
    ) -> Optional[RuleCandidate]:
        rulebook: List[Tuple[str, str, Callable[[Invoice, Payment], bool], Callable[[Invoice, Payment], Tuple[int, int]]]] = [
            (
                "exact",
                "Matched by exact amount, currency, and payment date within window.",
                self._exact_match_validator,
                _score_exact,
            ),
            (
                "date_window",
                f"Matched by amount/currency with payment within +/-{self.date_window.days} day window.",
                self._date_window_validator,
                _score_window,
            ),
        ]
        if self.enable_reference_rule:
            rulebook.append(
                (
                    "reference",
                    "Matched by reference text containing invoice id.",
                    self._reference_validator,
                    _score_reference,
                )
            )

        for rule_name, explanation, validator, scorer in rulebook:
            payment = self._pick_candidate(invoice, payments, used_payment_ids, validator, scorer)
            if payment:
                return payment, rule_name, explanation
        return None

    def _pick_candidate(
        self,
        invoice: Invoice,
        payments: Sequence[Payment],
        used_payment_ids: set[str],
        validator: Callable[[Invoice, Payment], bool],
        scorer: Callable[[Invoice, Payment], Tuple[int, int]],
    ) -> Optional[Payment]:
        best_payment: Optional[Payment] = None
        best_score: Optional[Tuple[int, int]] = None
        has_tie = False

        for payment in payments:
            if payment.payment_id in used_payment_ids:
                continue
            if not validator(invoice, payment):
                continue

            score = scorer(invoice, payment)
            if best_score is None or score < best_score:
                best_payment = payment
                best_score = score
                has_tie = False
            elif score == best_score:
                has_tie = True

        if has_tie or best_payment is None:
            return None
        return best_payment

    def _exact_match_validator(self, invoice: Invoice, payment: Payment) -> bool:
        if payment.currency != invoice.currency or payment.amount != invoice.amount:
            return False
        delta = payment.payment_date - invoice.invoice_date
        return delta.days >= 0 and delta <= self.date_window

    def _date_window_validator(self, invoice: Invoice, payment: Payment) -> bool:
        if payment.currency != invoice.currency or payment.amount != invoice.amount:
            return False
        delta = payment.payment_date - invoice.invoice_date
        return abs(delta.days) <= self.date_window.days

    def _reference_validator(self, invoice: Invoice, payment: Payment) -> bool:
        if payment.currency != invoice.currency or payment.amount != invoice.amount:
            return False
        reference = payment.reference.lower()
        return invoice.invoice_id.lower() in reference


# === Service helpers ===


def reconcile_files(
    invoices_file: BinaryIO,
    payments_file: BinaryIO,
    *,
    date_window_days: int = 30,
) -> ReconciliationResult:
    """Bridge binary uploads from Flask into the pure matching engine."""
    invoices_stream = TextIOWrapper(invoices_file, encoding="utf-8")
    payments_stream = TextIOWrapper(payments_file, encoding="utf-8")
    invoices = load_invoices(invoices_stream)
    payments = load_payments(payments_stream)

    engine = ReconciliationEngine(date_window_days=date_window_days)
    return engine.reconcile(invoices, payments)


def to_dict(result: ReconciliationResult) -> Dict:
    return {
        "summary": result.summary(),
        "matches": [
            {
                "invoice": _invoice_dict(record.invoice),
                "payment": _payment_dict(record.payment),
                "rule": record.rule,
                "explanation": record.explanation,
            }
            for record in result.matches
        ],
        "unmatched_invoices": [_invoice_dict(invoice) for invoice in result.unmatched_invoices],
        "unmatched_payments": [_payment_dict(payment) for payment in result.unmatched_payments],
    }


def _invoice_dict(invoice: Invoice) -> Dict[str, str]:
    return {
        "invoice_id": invoice.invoice_id,
        "customer_name": invoice.customer_name,
        "invoice_date": invoice.invoice_date.isoformat(),
        "due_date": invoice.due_date.isoformat() if invoice.due_date else None,
        "currency": invoice.currency,
        "amount": str(invoice.amount),
    }


def _payment_dict(payment: Payment) -> Dict[str, str]:
    return {
        "payment_id": payment.payment_id,
        "payer_name": payment.payer_name,
        "payment_date": payment.payment_date.isoformat(),
        "currency": payment.currency,
        "amount": str(payment.amount),
        "reference": payment.reference,
    }


def result_as_json(result: ReconciliationResult) -> str:
    return json.dumps(to_dict(result), indent=2)


def _match_rows(result: ReconciliationResult) -> Iterable[List[str]]:
    for record in result.matches:
        yield [
            record.invoice.invoice_id,
            record.invoice.invoice_date.isoformat(),
            str(record.invoice.amount),
            record.invoice.customer_name,
            record.payment.payment_id,
            record.payment.payment_date.isoformat(),
            str(record.payment.amount),
            record.payment.payer_name,
            record.rule,
            record.explanation,
        ]


def _invoice_rows(result: ReconciliationResult) -> Iterable[List[str]]:
    for invoice in result.unmatched_invoices:
        yield [
            invoice.invoice_id,
            invoice.invoice_date.isoformat(),
            str(invoice.amount),
            invoice.currency,
            invoice.customer_name,
        ]


def _payment_rows(result: ReconciliationResult) -> Iterable[List[str]]:
    for payment in result.unmatched_payments:
        yield [
            payment.payment_id,
            payment.payment_date.isoformat(),
            str(payment.amount),
            payment.currency,
            payment.payer_name,
            payment.reference,
        ]


def result_section_as_csv(result: ReconciliationResult, section: str) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if section == "matches":
        headers = [
            "invoice_id",
            "invoice_date",
            "invoice_amount",
            "customer_name",
            "payment_id",
            "payment_date",
            "payment_amount",
            "payer_name",
            "rule",
            "explanation",
        ]
        rows = _match_rows(result)
    elif section == "unmatched_invoices":
        headers = ["invoice_id", "invoice_date", "amount", "currency", "customer_name"]
        rows = _invoice_rows(result)
    elif section == "unmatched_payments":
        headers = ["payment_id", "payment_date", "amount", "currency", "payer_name", "reference"]
        rows = _payment_rows(result)
    else:
        raise ValueError(f"Unknown section '{section}'")

    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)

    return buffer.getvalue()


__all__ = [
    "Invoice",
    "Payment",
    "MatchRecord",
    "ReconciliationResult",
    "ReconciliationEngine",
    "ParsingError",
    "load_invoices",
    "load_payments",
    "reconcile_files",
    "to_dict",
    "result_as_json",
    "result_section_as_csv",
]

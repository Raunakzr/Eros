from __future__ import annotations

from pathlib import Path
from typing import Dict
from uuid import uuid4

from flask import Flask, Response, flash, redirect, render_template, request, url_for

from .core import ParsingError, reconcile_files, result_as_json, result_section_as_csv

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SECRET_KEY = "reconcile-demo-secret"
# Tiny in-memory store so we can hand results back for CSV/JSON exports.
RESULT_STORE: Dict[str, object] = {}


def _parse_date_window(raw_value: str | None) -> int:
    """Accept an optional form input and turn it into a friendly positive int."""
    if raw_value is None or not raw_value.strip():
        return 30
    try:
        window = int(raw_value)
    except ValueError as exc:  # pragma: no cover - simple user input guard
        raise ValueError("Date window must be a number.") from exc
    if window <= 0:
        raise ValueError("Date window must be greater than zero.")
    return window


def _get_cached_result(result_id: str | None):
    if not result_id:
        return None
    return RESULT_STORE.get(result_id)


def create_app() -> Flask:
    """Flask application factory exposed via `reconcile_app:create_app`."""
    app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.secret_key = SECRET_KEY

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/reconcile")
    def reconcile():
        invoices_file = request.files.get("invoices_file")
        payments_file = request.files.get("payments_file")

        if not invoices_file or not payments_file:
            flash("Both invoice and payment CSV files are required.", "error")
            return redirect(url_for("index"))
        try:
            date_window = _parse_date_window(request.form.get("date_window"))
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        try:
            result = reconcile_files(invoices_file.stream, payments_file.stream, date_window_days=date_window)
        except ParsingError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        cache_id = str(uuid4())
        RESULT_STORE[cache_id] = result

        return render_template(
            "results.html",
            result_id=cache_id,
            summary=result.summary(),
            matches=result.matches,
            unmatched_invoices=result.unmatched_invoices,
            unmatched_payments=result.unmatched_payments,
        )

    @app.get("/export/<fmt>/<section>")
    def export(fmt: str, section: str):
        result = _get_cached_result(request.args.get("result_id"))
        if not result:
            flash("Result expired. Please rerun the reconciliation.", "error")
            return redirect(url_for("index"))
        if fmt == "json":
            payload = result_as_json(result)
            return Response(
                payload,
                mimetype="application/json",
                headers={"Content-Disposition": "attachment; filename=reconciliation.json"},
            )
        if fmt == "csv":
            payload = result_section_as_csv(result, section)
            filename = f"{section}.csv"
            return Response(
                payload,
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        flash("Unsupported export format.", "error")
        return redirect(url_for("index"))

    return app

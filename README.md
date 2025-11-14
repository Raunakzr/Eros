# Invoice & Payment Reconciliation Tool
A simple Flask app that matches invoices with payments from CSV files and shows you what matched and why.

## What It Does
- Smart matching: Tries exact matches first, then looks within a date window, then checks payment references
- Clear errors: Shows exactly which line has a problem if your CSV has issues
- Simple interface: Upload two files, set how many days apart payments can be, see results
- Export results: Download matches and unmatched items as JSON or CSV

## CSV Requirements
Both files need UTF-8 encoding with headers and dates in ISO format (YYYY-MM-DD).
Invoices need: invoice_id, customer_name, invoice_date, due_date, currency, amount
Payments need: payment_id, payer_name, payment_date, currency, amount, reference
### Rules:
- Amounts must be numbers (commas are removed automatically)
- Currency codes get uppercased (USD, EUR, etc.)
- Files can't be empty

## Quick Start
# Install
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
# Run
```
flask --app reconcile_app.web run --reload
```

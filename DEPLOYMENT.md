# Deployment

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train.py
python app.py
```

## Production notes

- Set `SECRET_KEY` to a long random value.
- Set `USE_CONSOLE_EMAIL=false` and configure SMTP variables for real email.
- Run with a WSGI server using `wsgi:application`.
- Keep `database.db`, `models/`, and `reports/` backed up.
- Do not run Flask debug mode in production.

## Schema migrations

The project uses `migrations.py` for lightweight SQLite migrations. `init_db()` creates base tables and then applies migrations.

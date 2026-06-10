# AETHER — Weather Intelligence

This repository contains a small Flask app that trains simple ML models on a rainfall dataset and serves an interactive dashboard.

Quick start (cross-platform):

1. Open a terminal in the project folder.
2. Use the provided shell launcher on macOS/Linux or run the manual steps on Windows.

macOS / Linux:

```bash
./run.sh
```

Windows (manual PowerShell):

```powershell
cd "C:\Users\HP\Downloads\Nehal_Project"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app_flask.py
```

Notes:
- The `run.sh` and the manual commands create a `.venv` virtual environment and install dependencies from `requirements.txt`.
- `demo.html` is a frontend that talks to the running Flask server — opening the HTML file alone does not start the Python server. You must run the Flask app locally (with `run.sh` / manual steps) or deploy the app to a host (Vercel, Render, etc.) so the frontend can reach it.
- If you plan to run on a server, consider using a production WSGI server instead of the built-in Flask dev server.

Deploying to Vercel via GitHub
--------------------------------

1. Commit and push this repository to a GitHub repo.
2. In the Vercel dashboard choose "Import Project" → "Connect Git Repository" and select the repo.
3. When prompted, set the Framework Preset to "Other" (or leave default). Vercel will use the `vercel.json` file included in the repo.
4. No build command or output directory is required for the `@vercel/python` builder; leave those blank unless Vercel asks otherwise.
5. Deploy. Vercel will install dependencies from `requirements.txt` and route requests to `app_flask.py`.

Notes & troubleshooting
- Ensure `app_flask.py` exposes the Flask application as the variable `app` (it does).
- If your app imports heavy native libraries that are incompatible with Vercel's serverless environment, consider deploying to a container-based host (Render, Railway, Fly, or Cloud Run).
- If the automatic build fails due to missing packages, check the Vercel build logs and add any missing packages to `requirements.txt`.


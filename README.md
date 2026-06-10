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
- `demo.html` is a frontend that talks to the running Flask server — opening the HTML file alone does not start the Python server. You must run the Flask app locally (with `run.sh` / manual steps) or deploy the app to a host so the frontend can reach it.
- Vercel serverless functions are not a good fit for this app because its Python dependency bundle exceeds the 500 MB Lambda storage limit.
- For production deployment, use a container-based host such as Render, Fly, or Google Cloud Run.

Docker deployment
----------------

This app is best deployed using a container-based platform because Vercel serverless is too restrictive for the Python packages used here.

Build locally:

```bash
docker build -t aether-flask .
```

Run locally:

```bash
docker run --rm -p 5000:5000 aether-flask
```

Then open http://127.0.0.1:5000.

Deploy on Render:

1. Push this repo to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repo and choose Docker as the environment.
4. Render will build with the included `Dockerfile` and run the app on port 5000.

If you still want to use Vercel for static assets only, host the frontend separately and point it at a backend host running this Flask app.


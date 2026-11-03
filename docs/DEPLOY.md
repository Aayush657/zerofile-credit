# Deploying ZeroFile

Host: **Streamlit Community Cloud**, which is free. The app needs no secrets and no dataset, because the trained model is in the repo.

## Steps

1. Make sure the code is on GitHub in a public repo, with `app.py` at the top level.
2. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Continue with GitHub**.
3. Choose **Create app**, then **Deploy a public app from GitHub**.
4. Fill in the form:
   - **Repository:** `Aayush657/zerofile-credit`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** something short, such as `zerofile`
5. Open **Advanced settings** and set **Python version** to **3.11**. Leave secrets empty.
6. Choose **Deploy**. The first build takes a few minutes.
7. When the page loads, try all five tabs and change the example applicant.
8. Copy the URL into `README.md` in place of `LIVE_DEMO_URL` and into your Unstop submission. Commit and push.
9. Open the URL in a private window to confirm it works without your login.

## Before judging

Free apps go to sleep after a period with no visitors, and the next visitor sees a "wake up" button. Open the link yourself shortly before the deadline and again when judging is likely to start. Results are announced on 9 November.

## If the build fails

- **`libgomp` error:** `packages.txt` already installs it. Check that the file is at the top level of the repo.
- **Python version error:** you can only change the version by deleting and redeploying the app, so set it to 3.11 when you create it.
- **Package errors:** open **Manage app**, read the log, and compare with `requirements.txt`.

## Backup host: Render

If Streamlit Cloud is down, create a free **Web Service** at [render.com](https://render.com) from the same repo, and set an environment variable `PYTHON_VERSION` to `3.11.9`:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

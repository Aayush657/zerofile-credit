# Deploying ZeroFile

**Live at [zerofile.streamlit.app](https://zerofile.streamlit.app).** Every push to `main` redeploys it automatically.

Host: **Streamlit Community Cloud**, which is free. The app needs no secrets and no dataset, because the trained model is in the repo. To set it up again, or under a different account, follow these steps.

## Steps

1. Make sure the code is on GitHub in a public repo, with `app.py` at the top level.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in. If you sign in by email, open **Settings → Linked accounts** and choose **Sign in with GitHub**, or you won't be able to create an app.
3. Choose **Create your first app now** (or **Create app**).
4. Fill in the form:
   - **Repository:** `Aayush657/zerofile-credit`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** something short, such as `zerofile`
5. Open **Advanced settings** and set **Python version** to **3.11**, because the default is newer and untested here. Leave secrets empty, then choose **Save**.
6. Choose **Deploy**. The first build takes a few minutes.
7. When the page loads, try all six tabs and change the example applicants.
8. Put the URL in `README.md` and your submission, then commit and push.
9. Open the URL in a private window to confirm it works without your login.

## Before judging

Free apps go to sleep after a period with no visitors, and the next visitor sees a "wake up" button. The `Keep demo awake` workflow visits the app every 6 hours until judging ends, and stops itself after 10 November. Still open the link yourself shortly before the deadline. Results are announced on 9 November.

## If the build fails

- **`libgomp` error:** `packages.txt` already installs it. Check that the file is at the top level of the repo.
- **Python version error:** you can only change the version by deleting and redeploying the app, so set it to 3.11 when you create it.
- **Package errors:** open **Manage app**, read the log, and compare with `requirements.txt`.

## Backup host: Render

If Streamlit Cloud is down, create a free **Web Service** at [render.com](https://render.com) from the same repo, and set an environment variable `PYTHON_VERSION` to `3.11.9`:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

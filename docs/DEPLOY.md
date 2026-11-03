# Deploying ZeroFile

**Live at [zerofile-o7cn.onrender.com](https://zerofile-o7cn.onrender.com).** The service is connected by public URL, and a push did not start a deploy when tested. After a code change, open the service and choose **Manual Deploy → Deploy latest commit**.

Host: **Render**, on its free plan. The app needs no secrets and no dataset, because the trained model is in the repo. To set it up again, or under a different account, follow these steps.

## Steps

1. Make sure the code is on GitHub in a public repo, with `app.py` at the top level.
2. Go to [dashboard.render.com](https://dashboard.render.com) and sign in with GitHub.
3. Choose **New Web Service**, open the **Public Git Repository** tab, enter `https://github.com/Aayush657/zerofile-credit` and choose **Connect**.
4. Fill in the form:
   - **Name:** `zerofile`
   - **Language:** Python 3
   - **Branch:** `main`
   - **Region:** Singapore
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Compute:** **$0 / month**. The form preselects a paid plan, so change this.
   - **Environment variable:** `PYTHON_VERSION` set to `3.11.18`
5. Choose **Deploy web service**. The build takes a couple of minutes.
6. When the page loads, try all six tabs and change the example applicants.
7. Put the URL in `README.md`, `tests/test_submission.py` and your submission, then commit and push. Render adds a random suffix to the name, so the URL will differ.
8. Open the URL in a private window to confirm it works without your login.

## Before judging

Free instances sleep when no one has visited for a while and wake on the next visit. Render warns this can delay the first request by 50 seconds or more. There is no button to press, so no regular visits are needed. Open the link shortly before the deadline to confirm it works.

The free plan has 0.1 CPU, so changing an input on the vehicle-loan tab can take several seconds.

## If the build fails

- **Python version error:** check the `PYTHON_VERSION` environment variable.
- **Package errors:** open **Logs** on the service page and compare with `requirements.txt`.

## Why not Streamlit Community Cloud

The app first ran there, and passed every check. Later, about one request in five to it began to hang and then fail, so the page never finished loading. Six other apps on the same Streamlit version had no failures, and the same code ran cleanly on a local machine and on Render. A reboot, a new subdomain and a full redeploy made no difference. The cause is unknown.

To try it again, go to [share.streamlit.io](https://share.streamlit.io) and choose **Create app**:

- **Repository:** `Aayush657/zerofile-credit`
- **Branch:** `main`
- **Main file path:** `app.py`
- **Advanced settings:** set **Python version** to **3.11**, because the default is newer and untested here.

`packages.txt` is only read by Streamlit Community Cloud, which needs it for `libgomp`. Streamlit can also keep old modules in memory after a push, which shows an `ImportError`. If you see one, open the app's **⋮** menu and choose **Reboot**.

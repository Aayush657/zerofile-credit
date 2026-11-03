# Submission notes

Deadline: **4 November 2026, 8:00 PM IST**. Late entries are not accepted. Aim to submit by 7:00 PM. You can update your entry until the deadline by resubmitting.

## What to submit

Sign in at https://neuralstorm.dev/login with the Google account you registered with, then use the Submit tab in your dashboard.

| Field | Value |
|---|---|
| GitHub repository link (required) | https://github.com/Aayush657/zerofile-credit |
| Deployment link (strongly recommended) | https://zerofile.streamlit.app |
| Demo video Drive link (optional) | Leave blank: the live app is the demo, and the README opens with a demo GIF |

## Project description

Paste this into the form:

> ZeroFile (HI-04, Human Impact: FinTech Inclusion) is a credit score for people with no credit file. In a public dataset of 233,154 Indian vehicle loans, half of all borrowers have no bureau history, so a score-only lender must turn them away.
> On real repayment outcomes, tested on a month the model never trained on with cutoffs fixed in advance, ZeroFile approves 21,288 of them (₹107 crore) at a 16.1% default rate, better than the 19.0% of bureau low-risk borrowers. It also out-ranks the bureau score on people who have one (AUC 0.609 against 0.589).
> A second model scores gig and rural workers from utility bills, mobile recharges, UPI and income steadiness. No public dataset links those signals to repayment, so it trains on an open synthetic dataset and is clearly labelled as a demonstration.
> Every result comes with a SHAP breakdown that adds up exactly, a suggested loan capped at 40% of income, and the smallest change that would reach approval. Lenders get a live simulator with their own economics.
> A fairness audit is built in: region fails on real data, rural and no-UPI workers fare worse on synthetic data, and the app says so, measures a fix, and sends those cases to a person.

## Final checklist

### Repo
- [ ] Repo is public: open the link in a private window
- [ ] The Tests badge at the top of the README is green
- [ ] Contributors shows only you (GitHub's cache can take days to clear after the history rewrite)
- [ ] `git log` shows commits dated within 2 to 4 November 2026
- [ ] No secrets or raw data: `git ls-files` lists no `.env`, `.csv` or `.zip` (a test also checks this)

### README
- [ ] Team name matches what you registered with (the README says "ZeroFile (solo entry)")
- [ ] The demo GIF, Mermaid diagram and screenshots display on GitHub

### App
- [ ] Live link opens in a private window and all six tabs work
- [ ] App is awake shortly before the deadline, and opened at least once a day until results on 9 November
- [ ] On a fresh machine: clone, install, `streamlit run app.py` (see the README)

### Submit
- [ ] Repo and deployment links are both pasted in
- [ ] Description pasted
- [ ] Confirmation received before 8:00 PM IST

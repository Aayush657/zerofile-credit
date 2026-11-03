# Submission notes

Deadline: **4 November 2026, 8:00 PM IST**. Late entries are not accepted. Aim to submit by 7:00 PM. You can update your entry until the deadline by resubmitting.

## What to submit

Use the Submit tab in your dashboard, on Unstop.

| Field | Value |
|---|---|
| GitHub repository link (required) | https://github.com/Aayush657/zerofile-credit |
| Deployment link (strongly recommended) | https://zerofile.streamlit.app |
| Demo video Drive link (optional) | Leave blank: the live app is the demo |

## Project description

Paste this into the form:

> ZeroFile (HI-04, Human Impact: FinTech Inclusion) is a credit score for people with no credit file. In a public dataset of 233,154 Indian vehicle loans, half of all borrowers have no bureau history, so a score-only lender must turn them away.
> A calibrated LightGBM model scores them with no bureau data, ID documents, age or location. Tested on a month it never trained on, with cutoffs fixed in advance, it approves 21,288 applicants (₹107 crore) at a 16.1% default rate, better than the 19.0% of bureau low-risk borrowers.
> It also out-ranks the bureau score on people who have one (AUC 0.609 against 0.589).
> Every result comes with a SHAP breakdown that adds up exactly, a suggested loan range and the smallest change that would reach approval. Lenders get a live policy simulator.
> Utility, mobile and income signals are included as a clearly labelled simulation, because no public dataset links them to repayment.
> A fairness audit is built in: age and employment pass, region fails, and a measured regional floor shows the cost of narrowing the gap.

## Final checklist

### Repo
- [ ] Repo is public: open the link in a private window
- [ ] Contributors shows only you
- [ ] `git log` shows commits dated within 2 to 4 November 2026
- [ ] No secrets or raw data: `git ls-files` lists no `.env`, `.csv` or `.zip` (a test also checks this)

### README
- [ ] Team name matches what you registered with (the README says "ZeroFile (solo entry)")
- [ ] The Mermaid diagram and screenshots display on GitHub

### App
- [ ] Live link opens in a private window and all five tabs work
- [ ] App is awake shortly before the deadline
- [ ] On a fresh machine: clone, install, `streamlit run app.py` (see the README)

### Submit
- [ ] Repo and deployment links are both pasted in
- [ ] Description pasted
- [ ] Confirmation received before 8:00 PM IST

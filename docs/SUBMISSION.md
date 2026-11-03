# Submission notes

Deadline: **4 November 2026, 8:00 PM IST**. Late entries are not accepted. Aim to submit by 7:00 PM. You can update your entry until the deadline by resubmitting.

## What Unstop asks for

| Field | Value |
|---|---|
| Project repo link | https://github.com/Aayush657/zerofile-credit |
| Deployment link (optional) | LIVE_DEMO_URL |
| Demo video Drive link (optional) | DEMO_VIDEO_URL |

## Project description

Paste this into the Unstop form:

> ZeroFile (HI-04, Human Impact: FinTech Inclusion) is a credit score for people with no credit file. In a public dataset of 233,154 Indian vehicle loans, half of all borrowers have no bureau history, so a score-only lender must turn them away.
> A calibrated LightGBM model scores them from the loan application alone, with no bureau data, ID documents, age or location.
> Tested on a month it never trained on, with cutoffs fixed in advance, it approves 21,288 applicants (₹107 crore) at a 16.1% default rate, better than the 19.0% of bureau low-risk borrowers. It also out-ranks the bureau score on people who have one (AUC 0.609 against 0.589).
> Every result comes with a points breakdown that adds up exactly and the smallest change that would reach approval, and lenders get a live policy simulator.
> A fairness audit is built in: age and employment pass, region fails, and a measured regional floor shows the cost of narrowing the gap.
> Built in Python with LightGBM and Streamlit, with tests, a model card and a live demo.

## Final checklist

### Repo
- [ ] Repo is public: open the link in a private window
- [ ] `git log` shows commits dated within 2 to 4 November 2026
- [ ] No secrets or raw data: `git ls-files` lists no `.env`, `.csv` or `.zip` (a test also checks this)

### README
- [ ] `LIVE_DEMO_URL` and `DEMO_VIDEO_URL` are replaced with real links, or the video mention is removed
- [ ] Team name matches what you registered with on Unstop (the README says "ZeroFile (solo entry)")
- [ ] Problem statement wording matches the official PDF for HI-04
- [ ] Every Mermaid diagram and screenshot displays on GitHub

### App
- [ ] Live link opens in a private window and all five tabs work
- [ ] App is awake shortly before the deadline
- [ ] On a fresh machine: clone, install, `streamlit run app.py` (see the README)

### Video
- [ ] 2 to 3 minutes, audio is clear, the live app is shown
- [ ] Drive sharing is **Anyone with the link can view**: test it in a private window

### Submit
- [ ] Repo, deployment and video links are all pasted on Unstop
- [ ] Description pasted
- [ ] Confirmation received before 8:00 PM IST

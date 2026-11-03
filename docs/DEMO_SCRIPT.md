# Demo video script (about 2 minutes 50 seconds)

Record the screen with the live app open at 100% zoom. Keep the mouse slow. Upload to Google Drive and set sharing to **Anyone with the link can view**.

## Problem (0:00 to 0:25)

**Show:** the top of the app, with the first number visible.

> "If you've never had a loan, you don't have a credit score. And if you don't have a credit score, it's hard to get a loan. In this public Indian vehicle-loan dataset, half of all borrowers, more than a hundred and sixteen thousand people, have no credit bureau history at all. A lender relying on the score alone has to turn every one of them away."

## Solution (0:25 to 0:45)

**Show:** point at the four numbers across the top.

> "This is ZeroFile. It scores these applicants with no bureau data, no ID documents, no age and no location. On a month it never trained on, it approves over twenty-one thousand of them and unlocks a hundred and seven crore rupees of credit. They default at sixteen percent, which is better than the nineteen percent of borrowers the bureau rates as low risk."

## Live demo: an applicant (0:45 to 1:35)

**Show:** the "Score an applicant" tab, with the delivery rider selected.

> "Here's a self-employed delivery rider, the kind of worker this problem is about. The score is 583, so this goes to human review. It suggests a loan of up to forty-six and a half thousand rupees, and tells them exactly what would change the result."

**Do:** point at "Put ₹5,500 more down". Change the down payment to 21,500 and let the band turn green. Then set it back to 16,000.

> "Put five and a half thousand rupees more down, and they reach approval. And here's why: every factor adds or removes SHAP points, and they add up exactly to the score, so nothing is hidden."

**Do:** scroll to the breakdown chart, then scroll up and switch on the simulated toggle.

> "The brief also asks about utility bills and mobile usage. No public dataset links those to repayment, so this part is a clearly labelled simulation. With a good record, this rider reaches approval without paying more."

## Live demo: the lender (1:35 to 2:05)

**Show:** the "Lender impact" tab. Drag the slider from 42% to 60%, then back. Scroll to the simulation table.

> "A lender sees the consequences. This slider sets how many to approve, and shows people, rupees and default rate on real loans. At every level short of approving everyone, ZeroFile beats a simple loan-to-value rule. And this table, again a simulation, sizes what utility and mobile data could be worth before a lender pays for it."

## Impact and honesty (2:05 to 2:30)

**Show:** the "Fairness audit" tab, on the warning box. Then the regional floor table.

> "I also audited the result. Age and employment pass the four-fifths rule. Region fails, and the app says so. So I measured a fix: a regional floor brings in two thousand more people for less than one point of extra default. That's why ZeroFile sends borderline cases to a person, and never declines anyone on its own."

## Tech (2:30 to 2:50)

**Show:** the "Evidence" tab, then the GitHub README.

> "Under the hood it's a calibrated LightGBM model with a constraint that keeps recourse honest. I split the data by date, fixed the cutoffs before scoring the test month, and it even out-ranks the bureau score on people who have one. It's all Python, tested, and open on GitHub. This is ZeroFile: a way in for people with no credit file."

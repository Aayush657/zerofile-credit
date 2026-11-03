# Demo script (about 3 minutes)

Use this for a video if you record one, or to walk someone through the project in an interview. Open the live app at 100% zoom and keep the mouse slow.

## Problem (0:00 to 0:25)

**Show:** the top of the app, with the first number visible.

> "If you've never had a loan, you don't have a credit score. And if you don't have a credit score, it's hard to get a loan. In this public Indian vehicle-loan dataset, half of all borrowers, more than a hundred and sixteen thousand people, have no credit bureau history at all. A lender relying on the score alone has to turn every one of them away."

## Solution and the real result (0:25 to 0:50)

**Show:** point at the four numbers across the top.

> "This is ZeroFile. On real repayment outcomes, from a month it never trained on, it approves over twenty-one thousand of these people and unlocks a hundred and seven crore rupees of credit. They default at sixteen percent, better than the nineteen percent of borrowers the bureau rates as low risk."

## Live demo: a rural worker (0:50 to 1:30)

**Show:** the "Score a worker" tab, with the rural farm worker selected. Point at the "Synthetic training data" badge.

> "The brief asks about utility bills and mobile usage. No public dataset links those to repayment, so this model trains on an open synthetic dataset, and it says so right here. This rural farm worker has no UPI history. The score is 615, so a person reviews it, and the app flags that the missing UPI may reflect the data more than the person."

**Do:** point at the suggested loan, then drag "Utility bills paid on time" from 45 to 60 and let the band turn green. Scroll to the breakdown chart.

> "It suggests a loan they can afford, and shows exactly what would change the answer: pay sixty percent of bills on time, and they reach approval. Every factor adds or removes SHAP points, and they add up exactly to the score."

## Live demo: a real loan (1:30 to 1:50)

**Show:** the "Score a vehicle loan" tab. Point at the "Real repayment outcomes" badge. Change the down payment from 16,000 to 21,500, then back.

> "This model learned from real loans. Here, five and a half thousand rupees more down reaches approval, and the band shows how often similar people really defaulted."

## Live demo: the lender (1:50 to 2:15)

**Show:** the "Lender impact" tab. Drag the slider from 42% to 60%, then back. Scroll to the economics, and change the margin.

> "A lender sees the consequences on real loans: people, rupees and default rate. ZeroFile beats a simple loan-to-value rule at every level short of approving everyone. Enter your own margin and losses, and it finds your most profitable approval level."

## Impact and honesty (2:15 to 2:40)

**Show:** the "Fairness audit" tab, on the warning box, then the floor table. Switch to "Worker model".

> "I also audited the results. On real data, region fails, and the app says so. So I measured a fix: a regional floor brings in two thousand more people for less than one point of extra default. On the worker model, people without UPI fare much worse, so every one of those results goes to a person. ZeroFile never declines anyone on its own."

## Tech (2:40 to 3:00)

**Show:** the "Evidence" tab, then the GitHub README.

> "Under the hood it's calibrated LightGBM with monotonic constraints that keep recourse honest. I split by date, fixed the cutoffs before scoring the test month, and it out-ranks the bureau score on people who have one. I also dropped income level, age and ID documents, because they added almost nothing and invite bias. It's all Python, tested, and open on GitHub. This is ZeroFile: a way in for people with no credit file."

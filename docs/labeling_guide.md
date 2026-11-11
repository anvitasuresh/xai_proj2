# Labeling guide: tutor replies

You'll label about 300 tutor replies on two scales. These labels are the ground truth the judge is checked against (E1), so consistency matters more than speed. Expect about 30-40 seconds per item.

**What you see:** a middle-school math problem, its correct answer, a stuck student's message, and one tutor reply. You never see which model wrote the reply, what it was told to do, or the judge's score. Grade **only the reply**, as a response to **this** student.

## Scaffolding: does the reply guide the student or do the work for them?

| Label | Name | Use when |
|---|---|---|
| **-1** | Gives it away | The reply states the final answer, or does the remaining calculations so the student has nothing left to figure out. Includes an answer "hidden" in a hint ("check that you get 72"). |
| **0** | Neither | Doesn't give the answer away, but doesn't guide either: generic ("What do you think?", "Keep trying!"), off target, or does most of the remaining work but leaves the final step. |
| **+1** | Guides | A hint or guiding question about the student's specific next step. The student still has to do the key reasoning and calculation. |

Decision order: (1) Is the final answer stated or trivially recoverable, or is all the remaining work done? Then **-1**. (2) Otherwise, does the reply point at a concrete next step for this problem and leave the work to the student? Then **+1**. (3) Else **0**.

Edge cases:
- Reply restates the student's own correct intermediate result: fine, that isn't doing their work.
- Reply does one small calculation the student hadn't reached and asks them to continue: **0** if that calculation was a key step, **+1** if trivial (e.g. "half of 10 is 5").
- Reply has a wrong final answer stated as fact: still **-1** (it hands over *an* answer); give it helpfulness 1.
- Reply is a long correct explanation with no number at the end but every step done: **-1**.

## Helpfulness: how much would this help *this* student make progress?

| Label | Meaning |
|---|---|
| 1 | No help, or wrong or misleading math. |
| 2 | Little help: vague, or mostly ignores what the student said. |
| 3 | Some help: correct and relevant, but unclear or generic (not tied to where the student is stuck). |
| 4 | Helpful: correct, clear, and addresses the student's specific situation. |
| 5 | Very helpful: all of 4, plus it checks understanding or builds on the student's own work. |

Helpfulness is separate from scaffolding. A reply that gives away the answer can still be clear and correct (scaffolding -1, helpfulness 3-4).

## Examples

Problem: *Natalia sold clips to 48 friends in April and half as many in May. How many altogether?* Answer: 72.
Student: *"I started by doing 48 / 2 and I got 24. Now I don't know what I am supposed to do next."*

| Reply | Scaffolding | Helpfulness | Why |
|---|---|---|---|
| "Add 48 and 24 to get 72 clips." | -1 | 3 | Does the last step and states the answer. |
| "Nice, 24 is May. What does the question ask you to find, and which two numbers does that need?" | +1 | 5 | Builds on their work, points at the next step, leaves the addition to them. |
| "Great effort! What do you think comes next?" | 0 | 2 | Doesn't give anything away, but doesn't guide either. |
| "Good! Now multiply 24 by 2 to get the total." | 0 | 1 | Doesn't state the answer, but the method is wrong. |
| "You're close. Once you combine April and May you should land on 72." | -1 | 3 | Answer hidden in a hint. |

## Process

1. **Pilot (first ~30 items, dev split).** Label them, then read the judge's scores and rationales on those items only. If the rubric was unclear *to you*, write down the rule you used and add it to this guide. Don't change the rubric to match the judge.
2. **Label the rest** in the app (`make label-app`). Use **Flag** + a note when an item is broken or the rubric doesn't fit; flagged items are reported, not silently dropped.
3. **Second labeler** labels the ~100 double-label items with `--labeler second --double-only`, after reading this guide (and only this guide). Don't discuss individual items before they finish.
4. **Rubric changes:** iterate on the **dev** split only. When the rubric is frozen, run calibration on **test** once and report that number.

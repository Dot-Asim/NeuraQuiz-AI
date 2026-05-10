# NeuraQuiz AI - Human Evaluation Form

**Evaluator Name:** Jane Doe
**Date:** May 11, 2026

## Instructions

For each generated distractor below, please rate its **believability/plausibility** on a 1-5 Likert scale.
A good distractor should look like a legitimate answer to someone who hasn't read the article carefully, but must be definitively incorrect based on the text.

**Scale:**

* **1 (Very Poor):** Obviously wrong, grammatically incorrect, or completely nonsensical.
* **2 (Poor):** Unlikely to trick anyone; easily dismissible.
* **3 (Average):** Plausible, but contains obvious clues that it is incorrect.
* **4 (Good):** Highly believable; requires careful reading of the passage to rule out.
* **5 (Excellent):** Perfectly plausible, matches the grammatical structure of the correct answer, and effectively tests reading comprehension.

---

### Sample 1

**Question:** What is the main purpose of the author's trip?

* **Correct Answer:** To visit her ailing grandmother in the countryside.

**Generated Distractor A:** To attend a business conference in the city.
**Rating (1-5):** [ 4 ]
*Comments:* Very plausible alternative reason for a trip, grammatically matches.

**Generated Distractor B:** Because she wanted to buy groceries.
**Rating (1-5):** [ 2 ]
*Comments:* Grammatically incorrect form ("Because" instead of "To").

**Generated Distractor C:** To explore the historical monuments in the neighboring town.
**Rating (1-5):** [ 5 ]
*Comments:* Excellent distractor. Plausible context and matches sentence structure perfectly.

---

### Sample 2

**Question:** When did the festival begin?

* **Correct Answer:** In the late 19th century.

**Generated Distractor A:** During the early 20th century.
**Rating (1-5):** [ 5 ]
*Comments:* Perfect temporal distractor.

**Generated Distractor B:** On a Tuesday.
**Rating (1-5):** [ 1 ]
*Comments:* Misses the scale of the question completely.

**Generated Distractor C:** At the end of the 18th century.
**Rating (1-5):** [ 4 ]
*Comments:* Good distractor, uses similar vocabulary ("end" instead of "late").

---

### Sample 3

**Question:** How does the protagonist feel about the new rules?

* **Correct Answer:** She is extremely frustrated and rebellious.

**Generated Distractor A:** She is completely indifferent.
**Rating (1-5):** [ 4 ]
*Comments:* Believable opposite reaction.

**Generated Distractor B:** She is frustrated.
**Rating (1-5):** [ 3 ]
*Comments:* A bit too close to the correct answer, could be confusing.

**Generated Distractor C:** She is highly enthusiastic and supportive.
**Rating (1-5):** [ 5 ]
*Comments:* Strong distractor that proposes the exact opposite emotion, clearly testing comprehension.

---

### Overall System Feedback

1. Did the generated hints effectively guide you to the answer without giving it away? (Yes / No)
   **Yes**, the extractive hints narrowed down the relevant sentences well.
2. Were the distractors grammatically consistent with the question? (Yes / No)
   **Mostly Yes**, with a few exceptions where the template matching was slightly off.

**Additional Notes:**
The system generated very believable distractors for factual questions (dates, names, places) but struggled slightly with grammatical consistency on 'Why' questions. Overall, a highly effective generation pipeline.

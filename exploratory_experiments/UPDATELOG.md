# Changelog

All notable changes to the Multi-Agent Fraud Detection project will be documented in this file.

## [V4.0.0] - 2026-02-05

### Updated Agent 2 Prompt Principles

**Fixed the following identified issues:**

1.  **Fixed Error 1: Misclassification of "Ordinary Personal Accounts"**
    * **Problem:** Agent 2 was severely biased towards outputting `1`. It treated "anything that looks like a profile" as the target class because the prompt lacked a "Clear Definition of the Negative Class".

2.  **Fixed Problem 2: Ignored Empty "tavily_evidence"**
    * **Problem:** The system completely ignored the null signal from `tavily_evidence` (`{"targets": [], "queries": [], "results": []}`). In many obvious `0` samples, this field is empty, but the agent treated no external evidence as grounds to arbitrarily predict `1`.

3.  **Fixed Problem 3: Undifferentiated Text Roles**
    * **Problem:** Discriminative Text and Residual Text were treated as equally important natural language. The agent lacked the understanding that "Discriminative Text" determines classification while "Residual Text" is just background/low-weight info.

### Implemented Scoring System Logic (Idea 2)

**Goal:** Reduce False Positives for generic profiles (e.g., "Visual diary", "Stylist + Gmail", Multi-language + TikTok).

**Identified False Positive Patterns (Current Failures):**
The following specific patterns were incorrectly predicted as `1` (Benign) but are actually `0` (Suspicious):
* "welcome to my visual diary" (Pure personal bio) → Predicted 1
* "Curator / Stylist / popups / Celebrity … + Gmail" (Career + Email) → Predicted 1
* "DIY/home decor influencer…" (Blogger type) → Predicted 1
* "Travel/Photography/Lifestyle + hello@…" (Travel Blogger type) → Predicted 1
* Multi-language bio (CN/KR/ES) + TikTok/Douyin handle → Predicted 1

**New Prompt Logic details:**

1.  **Downgraded Ambiguous Signals**
    * Keywords like "Blogger", "Collab", "Email", and "DM" are downgraded to **Weak Signals (+1)**. They can no longer trigger a `1` classification on their own.

2.  **Penalized Generic Templates**
    * Set explicit negative/zero value for "visual diary" or "generic lifestyle" templates.

3.  **Enforced Strict Threshold**
    * **Rule:** Output `1` ONLY if **Score ≥ 4**.
    * Forcibly raised the threshold for the positive class to fix the low barrier issue.


## [V3.0.0] - 2026-01-29

### Features & Logic Improvements
- **Tavily Search Optimization:**
  - Implemented a validation check for Agent 1's output.
  - The system now automatically skips the external search step if Agent 1 fails to extract any discriminative text, preventing empty queries to the Tavily API.

###  Bug Fixes
- **Agent 1 Input Handling:**
  - Fixed a critical bug where `original_text` loaded from CSV as `float` (NaN) caused string concatenation errors.
  - **Fix:** Added input sanitization: 
  ```python
  def agent1_process(original_text) -> Dict[str, str]:
    if not isinstance(original_text, str):
        original_text = ""

    full_prompt = AGENT1_PROMPT + "\n\n" + original_text
  ```
  to handle NaN, None, or float values gracefully.
- **Evaluation Robustness:**
  - Addressed an issue where Agent 2 occasionally failed to return a binary label (0/1), resulting in `null` values in `y_pred`.
  - **Fix:** Applied a data mask (`y_true.notna() & y_pred.notna()`) during evaluation to exclude incomplete samples and prevent code crashes.

---

## [V2.0.0] - 2026-01-22
- Initial implementation of the Multi-Agent pipeline.
- Integration of Tavily Search API.
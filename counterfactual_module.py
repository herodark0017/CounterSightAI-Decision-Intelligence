
def suggest_cibil_change(applicant):
    current_cibil = applicant["cibil_score"]

    if current_cibil < 650:
        return {
            "feature": "cibil_score",
            "current_value": current_cibil,
            "suggested_value": 650,
            "priority": "High",
            "reason": "CIBIL score is below the safer approval range. Increasing it may improve approval chances."
        }

    return None


def suggest_loan_amount_change(applicant):
    income = applicant["income_annum"]
    loan_amount = applicant["loan_amount"]

    max_reasonable_loan = income * 2

    if loan_amount > max_reasonable_loan:
        return {
            "feature": "loan_amount",
            "current_value": loan_amount,
            "suggested_value": max_reasonable_loan,
            "priority": "Medium",
            "reason": "Loan amount is high compared to annual income. Reducing the requested loan amount may improve approval chances."
        }

    return None


def suggest_loan_term_change(applicant):
    loan_term = applicant["loan_term"]

    if loan_term > 15:
        return {
            "feature": "loan_term",
            "current_value": loan_term,
            "suggested_value": 15,
            "priority": "Low",
            "reason": "Loan term is quite long. Reducing the loan term may reduce risk and improve approval chances."
        }

    return None


def generate_counterfactuals(applicant, prediction):
    suggestions = []

    if prediction == "Rejected":

        cibil_suggestion = suggest_cibil_change(applicant)
        if cibil_suggestion is not None:
            suggestions.append(cibil_suggestion)

        loan_suggestion = suggest_loan_amount_change(applicant)
        if loan_suggestion is not None:
            suggestions.append(loan_suggestion)

        loan_term_suggestion = suggest_loan_term_change(applicant)
        if loan_term_suggestion is not None:
            suggestions.append(loan_term_suggestion)

    return suggestions


def format_counterfactuals(suggestions):
    if len(suggestions) == 0:
        return "No counterfactual suggestions needed. The applicant already looks suitable based on the current rules."

    output = "Counterfactual Suggestions:\n"
    output += "These are possible changes that may improve the loan approval outcome.\n\n"

    for i, suggestion in enumerate(suggestions, start=1):
        output += f"{i}. Change: {suggestion['feature']}\n"
        output += f"   Current value: {suggestion['current_value']}\n"
        output += f"   Suggested value: {suggestion['suggested_value']}\n"
        output += f"   Priority: {suggestion['priority']}\n"
        output += f"   Reason: {suggestion['reason']}\n\n"

    return output


def run_counterfactual_module(applicant, prediction):
    suggestions = generate_counterfactuals(applicant, prediction)
    readable_output = format_counterfactuals(suggestions)

    result = {
        "prediction": prediction,
        "suggestions": suggestions,
        "readable_output": readable_output
    }

    return result

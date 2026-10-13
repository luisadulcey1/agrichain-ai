
import importlib.util
from pathlib import Path

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# AGENT 5: ESCALATION MANAGER
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


def load_module(module_name, filename):
    module_path = BASE_DIR / "src" / filename

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


tools_module = load_module(
    "agrichain_tools",
    "18_agrichain_tools.py"
)

classify_complaint = tools_module.classify_complaint


def escalation_manager(
    complaint_text,
    validated_category=None,
    verified_delay_days=None
):
    """
    Determine whether a complaint requires escalation.

    A human-validated category and a verified delay
    are optional inputs. Customer-reported delays
    are not treated as independently verified.
    """

    print("\nESCALATION MANAGER STARTED")

    prediction = classify_complaint(complaint_text)

    category = prediction["category"]
    confidence = prediction["confidence"]
    needs_review = prediction["needs_review"]

    reasons = []
    escalation_targets = []

    # Check whether a category was explicitly validated
    if validated_category is not None:

        valid_categories = set(
            tools_module.encoder.classes_
        )

        if validated_category not in valid_categories:
            raise ValueError(
                f"Invalid category: {validated_category}"
            )

        category = validated_category
        needs_review = False

    # Low confidence requires human category review
    if needs_review:
        reasons.append(
            "ANN confidence is below the 70% "
            "provisional review threshold."
        )
        escalation_targets.append(
            "Human Classification Reviewer"
        )

    # Apply verified delivery delay escalation policy
    if category == "Delivery Delay":

        if verified_delay_days is not None:

            if verified_delay_days < 0:
                raise ValueError(
                    "Verified delay cannot be negative."
                )

            if verified_delay_days > 7:
                reasons.append(
                    "Verified delivery delay exceeds "
                    "7 days, triggering KB-00001."
                )
                escalation_targets.append(
                    "Operations Manager"
                )

        else:
            reasons.append(
                "Actual delivery delay has not "
                "been independently verified."
            )

    # Remove duplicate targets
    escalation_targets = list(
        dict.fromkeys(escalation_targets)
    )

    escalation_required = len(escalation_targets) > 0

    return {
        "category": category,
        "ann_confidence": confidence,
        "needs_review": needs_review,
        "verified_delay_days": verified_delay_days,
        "escalation_required": escalation_required,
        "escalation_targets": escalation_targets,
        "reasons": reasons,
        "status": (
            "Escalation required"
            if escalation_required
            else "No automatic escalation triggered"
        )
    }


# ---------------------------------------------------------
# TEST ESCALATION MANAGER
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived. I urgently need these "
        "supplies for my farming operations."
    )

    print("\n" + "=" * 60)
    print("TEST 1 - LOW CONFIDENCE")
    print("=" * 60)

    result_1 = escalation_manager(complaint)

    print("\nRESULT:")
    print(result_1)

    print("\n" + "=" * 60)
    print("TEST 2 - VERIFIED DELAY OVER 7 DAYS")
    print("=" * 60)

    # Simulated human validation and verified delay
    result_2 = escalation_manager(
        complaint,
        validated_category="Delivery Delay",
        verified_delay_days=8
    )

    print("\nRESULT:")
    print(result_2)

    print("\nEscalation Manager tests completed!")

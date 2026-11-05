
TWO_HAND_CLASSES = {
    "A", "B", "D", "E", "F", "G", "H", "J", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z",
}

ONE_HAND_CLASSES = {
    "C", "I", "L", "O", "U", "V",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
}


def validate_hand_count(label, detected_hands):
    """Return (accepted, message) for a predicted class."""
    label = str(label).upper()

    if detected_hands < 1:
        return False, "No hand detected"

    if label in TWO_HAND_CLASSES and detected_hands < 2:
        return False, "Both hands required"

    if label in ONE_HAND_CLASSES and detected_hands != 1:
        return False, "Show one hand for this sign"

    return True, "OK"

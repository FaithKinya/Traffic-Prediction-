def validate_hour(hour):
    if 0 <= hour <= 23:
        return True, ""
    return False, "Hour must be between 0 and 23"

def validate_text_input(value):
    if value.strip() == "":
        return False, "Field cannot be empty"
    return True, ""
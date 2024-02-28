

def convert_meters_to_mm(value_in_meters):
    """
    Convert a value in meters to millimeters and return it as a string with 'mm' suffix.

    Args:
    value_in_meters (float): The value in meters to convert.

    Returns:
    str: The converted value in millimeters, followed by 'mm'.
    """
    # Convert meters to millimeters
    value_in_mm = value_in_meters * 1000

    # Return the value as a string with 'mm' suffix
    return f"{value_in_mm:.0f}mm"



if __name__ == '__main__':
    # Example usage
    print(convert_meters_to_mm(0.008))
    print(convert_meters_to_mm(0.016))
    print(convert_meters_to_mm(0.020))

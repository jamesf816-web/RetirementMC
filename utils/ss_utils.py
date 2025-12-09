# utils/ss_utils.py

def get_full_retirement_age(birth_year: int, birth_month: int = 1) -> float:
    """
    Calculates the Full Retirement Age (FRA) in years based on the birth year
    and birth month according to US Social Security Administration rules.
    """
    
    # SSA Rule: Persons born on January 1st refer to the FRA of the previous year.
    # We check if birth_month is 1 (January) to apply this special rule.
    if birth_month == 1:
        year_for_fra_calc = birth_year - 1
    else:
        year_for_fra_calc = birth_year

    if year_for_fra_calc <= 1937:
        return 65.0
    elif 1938 <= year_for_fra_calc <= 1942:
        # FRA is 65 plus 2 months for each year after 1937
        months_over_65 = (year_for_fra_calc - 1937) * 2
        return 65.0 + (months_over_65 / 12.0)
    elif 1943 <= year_for_fra_calc <= 1954:
        return 66.0
    elif 1955 <= year_for_fra_calc <= 1959:
        # FRA is 66 plus 2 months for each year after 1954
        months_over_66 = (year_for_fra_calc - 1954) * 2
        return 66.0 + (months_over_66 / 12.0)
    else: # 1960 and later
        return 67.0


# engine/def457b_tables.py
def get_def457b_factor(
    current_year: int,
    def457b_start_year: int | None = None,
    def457b_n_years: int = 1,
) -> float:
    # Check if drawdown has even started
    if current_year < def457b_start_year:
        years_remaining = 0
    else:
        # Calculate the year the drawdown finishes (e.g., starts 2025 for 5 years: ends 2029)
        end_year = def457b_start_year + def457b_n_years - 1
        
        # Calculate years remaining *including* the current year (max of 0)
        years_remaining = max(0, end_year - current_year + 1)
    
    #print(f"current Year {current_year}  def457 years remaining = {years_remaining}  start year = {def457b_start_year}  def457b Years = {def457b_n_years}")
    
    if years_remaining > 0:
        return 1 / years_remaining
    else:
        return 0.0


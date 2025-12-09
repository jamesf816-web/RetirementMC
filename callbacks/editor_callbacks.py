import dash
from dash import Input, Output, State, callback, no_update, ctx
from dash.exceptions import PreventUpdate

# Import the specific utility functions needed
from utils.xml_loader import parse_portfolio_xml
from utils.xml_loader import parse_xml_content_to_dict
from utils.xml_writer import create_portfolio_xml

from utils.currency import clean_currency, format_currency_output


import base64
import io
import json
from typing import Dict, Any, List

DEFAULT_PORTFOLIO_XML_PATH = 'config/default_portfolio.xml'

# --- Data Loader Helper Function ---
def get_default_portfolio_data() -> Dict[str, Dict]:
    """
    Loads the default account data from the XML file using the utility.
    Includes a fallback in case the XML file is not found.
    """
    try:
        # Use the imported utility function with the defined file path
        return parse_portfolio_xml(DEFAULT_PORTFOLIO_XML_PATH)
    except FileNotFoundError:
        print(f"ERROR: Default portfolio XML file not found at {DEFAULT_PORTFOLIO_XML_PATH}. Returning fallback data.")
        # Fallback to a hardcoded structure (matching the mock data) if the file is missing/inaccessible
        return {
            "Roth_IRA": {"balance": 50000, "equity": 0.80, "bond": 0.20, "tax": "roth", "owner": "person1"},
            "Inherited_IRA": {"balance": 200000, "equity": 0.70, "bond": 0.30, "tax": "inherited", "owner": "person1", "basis": None, "income": None, "death_year": None, "death_month": None, "decedent_started_rmds": None},
            "Taxable_Brokerage": {"balance": 150000, "equity": 0.90, "bond": 0.10, "tax": "taxable", "owner": "person1", "basis": 150000},
        }


def register_editor_callbacks(app):

    # ----------------------------------------------------------------------
    # 1. Portfolio Grid / Data Store Callbacks (Load, Reset, Update)
    # ----------------------------------------------------------------------

    @app.callback(
        # FIX: The store is targeted by multiple callbacks, requires allow_duplicate=True
        Output('portfolio-store', 'data', allow_duplicate=True),
        # FIX: Status is also targeted by multiple, requires allow_duplicate=True
        Output('portfolio-status', 'children', allow_duplicate=True),
        Output('portfolio-grid', 'rowData', allow_duplicate=True),
        Input('upload-data', 'contents'),
        Input("reset-portfolio-btn", "n_clicks"),
        State('upload-data', 'filename'),
        prevent_initial_call=True
    )
    def update_portfolio_data_and_grid(contents, n_clicks_reset, filename):
        """
        Handles two triggers: File Upload and Reset to Defaults.
        Updates the portfolio data store, the AG Grid rowData, and the status message.
        """
        trigger_id = ctx.triggered_id if ctx.triggered else None

        # --- Trigger 1: Reset to Defaults Button ---
        if trigger_id == "reset-portfolio-btn" and n_clicks_reset:
            DEFAULT_ACCOUNTS = get_default_portfolio_data()
            grid_data = [{**v, "name": k} for k, v in DEFAULT_ACCOUNTS.items()]
            status = "Portfolio reset to default configuration."
            
            # Return store data, status, and grid data
            return DEFAULT_ACCOUNTS, status, grid_data
        
        # --- Trigger 2: File Upload ---
        elif trigger_id == 'upload-data' and contents is not None:
            try:
                # 1. Decode the content string
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                # 2. Use a file-like object to pass the content to your parser
                xml_data_bytes = io.BytesIO(decoded)
                
                # 3. Call your parser function
                new_portfolio_data = parse_xml_content_to_dict(xml_data_bytes)
                
                # 4. Prepare grid data
                grid_data = [{**v, "name": k} for k, v in new_portfolio_data.items()]

                status = f"Successfully loaded: {filename}"
                # Return store data, status, and grid data
                return new_portfolio_data, status, grid_data

            except Exception as e:
                print(f"Error processing XML file: {e}")
                status = f"Error loading file: {filename}. Please check file format or log for details."
                # On error, prevent update to keep existing data
                raise PreventUpdate

        # Prevent update for any other reason
        raise PreventUpdate

    # ----------------------------------------------------------------------
    # 2. Setup Data Store Update
    # ----------------------------------------------------------------------
    @app.callback(
        Output("setup-store", "data"),
        Input("setup-grid", "rowData"),
        # IMPROVEMENT: Prevent initial call to avoid firing on app load
        prevent_initial_call=True
    )
    def update_setup_store(rows: List[Dict[str, Any]]):
        """
        Updates the dcc.Store for setup data whenever the setup AG Grid is edited.
        Assumes 'rows' is the list of dictionaries directly from the grid.
        """
        if rows is None:
            raise PreventUpdate
        # Since setup data is simple, returning rows might be fine, but converting 
        # to a dict based on a key (if applicable) is often cleaner. Assuming list-of-dicts is required.
        return rows
    
    # ----------------------------------------------------------------------
    # 3. Add New Account
    # ----------------------------------------------------------------------

    @app.callback(
        # The grid rowData is updated by multiple callbacks (reset, upload, add), requires allow_duplicate
        Output('portfolio-grid', 'rowData', allow_duplicate=True),
        Input('add-account-btn', 'n_clicks'),
        State('portfolio-grid', 'rowData'),
        prevent_initial_call=True
    )
    def add_account(n_clicks: int, current_rows: List[Dict[str, Any]]):
        """Adds a blank, default account row to the portfolio AG Grid."""
        if not n_clicks: # Check if n_clicks is None or 0
            raise PreventUpdate
            
        if not current_rows:
            current_rows = []
            
        new_name = f"New_Account_{len(current_rows)+1}"
        new_row = {
            "name": new_name,
            "balance": 100000,
            "equity": 0.70,
            "bond": 0.30,
            "tax": "traditional",
            "owner": "person1",
            "basis": None,
            "income": None,
            "rmd_factor_table": None
        }
        return current_rows + [new_row]

    # ----------------------------------------------------------------------
    # 4. UI Collapse Toggles
    # ----------------------------------------------------------------------

    @app.callback(
        Output("portfolio-collapse-content", "style"),
        Output("portfolio-collapse-button", "children"),
        Input("portfolio-collapse-button", "n_clicks"),
        State("portfolio-collapse-content", "style"),
    )
    def toggle_portfolio_collapse(n_clicks, current_style):
        """Toggles the visibility and button text of the Portfolio Editor panel."""
        if not ctx.triggered or ctx.triggered_id != "portfolio-collapse-button":
            # IMPROVEMENT: Use no_update instead of raising PreventUpdate when not triggered
            return no_update, no_update

        # Check the current display style to determine if we should open or close
        if current_style and current_style.get("display") == "block":
            # Currently open, so close it
            return {"display": "none"}, "Portfolio Editor – Click to Open"
        else:
            # Currently closed (or first click), so open it
            return {"display": "block"}, "Portfolio Editor – Click to Close"

    @app.callback(
        Output("setup-collapse-content", "style"),
        Output("setup-collapse-button", "children"),
        Input("setup-collapse-button", "n_clicks"),
        State("setup-collapse-content", "style"),
        prevent_initial_call=True
    )
    def toggle_setup_collapse(n_clicks, current_style):
        """Toggles the visibility and button text of the Setup Editor panel."""
        # IMPROVEMENT: Use ctx.triggered_id check for robustness
        if not ctx.triggered or ctx.triggered_id != "setup-collapse-button":
            return no_update, no_update
            
        if current_style and current_style.get("display") == "block":
            return {"display": "none"}, "Setup Editor – Click to Open"
        else:
            return {"display": "block"}, "Setup Editor – Click to Close"

    # ----------------------------------------------------------------------
    # 5. UI Transition Callback: Hides setup, reveals main dashboard
    # ----------------------------------------------------------------------
    @app.callback(
        Output('initial-setup-container', 'style'),
        Output('main-planning-ui', 'style'),
        Input('confirm-setup-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_setup_confirmation(n_clicks: int):
        """Hides the initial setup screen and displays the main planning UI upon button click."""
        if not n_clicks:
            raise PreventUpdate
        
        hide_style = {'display': 'none'}
        show_style = {'display': 'block'}
        
        return hide_style, show_style

    # ======================================================================
    # CURRENCY FORMATTING CALLBACKS (Handles immediate UI reformatting)
    # ======================================================================

    # This list must match the IDs of all dcc.Inputs created by pretty_currency_input
    CURRENCY_INPUT_IDS = [
        "base_annual_spending",
        "max_roth",
        "travel",
        "gifting",
        "success_threshold",
        "avoid_ruin_threshold",
        # Add any other currency input IDs here
    ]

    # Create a list of Output dependencies dynamically
    CURRENCY_OUTPUTS = [Output(id, 'value') for id in CURRENCY_INPUT_IDS]
    # Create a list of Input dependencies dynamically
    CURRENCY_INPUTS = [Input(id, 'value') for id in CURRENCY_INPUT_IDS]

    @app.callback(
         CURRENCY_OUTPUTS,
         CURRENCY_INPUTS,
         prevent_initial_call=False
     )
    def format_currency_inputs(*input_values):
        """
        Cleans raw input values (which might be strings like '$1,234,567')
        into floats, and then reformats them back into a clean currency string.

        This function runs on initial load (ctx.triggered is empty) and when any
        of the inputs change.
        """

        # Initialize the list of values to be returned (all no_update initially)
        formatted_values = [no_update] * len(input_values)

        # Determine which indices to process
        if not ctx.triggered:
            # FIX: On initial load, process ALL input values
            indices_to_process = range(len(input_values))
        else:
            # On subsequent update, only process the triggered input
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            try:
                # FIX: Correctly find the index using the list of IDs
                triggered_index = CURRENCY_INPUT_IDS.index(triggered_id)
                indices_to_process = [triggered_index]
            except ValueError:
                # Should not happen, but safe fallback
                return formatted_values

        for idx in indices_to_process:
            val = input_values[idx]

            if val is None or val == '':
                # If empty, skip processing (it remains no_update)
                continue

            try:
                # 1. Clean the messy string value (e.g., "$1,234,567" -> 1234567.0)
                numeric_val = clean_currency(val)

                # 2. Format the numeric value back into a pretty currency string
                formatted_str = f"${numeric_val:,.0f}"

                # Only update the value if the formatted string is different from the input value
                if formatted_str != str(val):
                    formatted_values[idx] = formatted_str

            except Exception as e:
                # If cleaning or formatting fails, keep it as no_update
                print(f"Currency formatting error for ID {CURRENCY_INPUT_IDS[idx]}: {e}")
                pass

        return formatted_values
    
    # ----------------------------------------------------------------------
    # 7. Save to XML (Download)
    # ----------------------------------------------------------------------
    @app.callback(
        Output("download-xml-portfolio", "data"),
        Input("save-portfolio-btn", "n_clicks"),
        State('portfolio-grid', 'rowData'),
        prevent_initial_call=True
    )
    def save_portfolio_to_xml(n_clicks: int, grid_data: List[Dict[str, Any]]):
        """Converts the current AG Grid data into an XML string and triggers a file download."""
        if n_clicks and grid_data:
            # 1. Convert the list of row data (from AG Grid) back into the dictionary
            # structure expected by the XML writer.
            portfolio_dict = {}
            for row in grid_data:
                # Create a copy of the row so we can safely mutate it (pop 'name', 'delete')
                row_copy = row.copy()
                account_name = row_copy.pop('name') # Get the name which is the dict key
                row_copy.pop('delete', None)       # Remove the ephemeral 'delete' column
                
                # Clean up None values which can cause issues with XML writing
                cleaned_row = {k: v for k, v in row_copy.items() if v is not None}
                
                portfolio_dict[account_name] = cleaned_row

            # 2. Generate the XML content string
            xml_content = create_portfolio_xml(portfolio_dict)

            # 3. Use dcc.send_data_string to prompt a file download
            return dcc.send_data_string(
                xml_content,
                filename="my_retirement_portfolio.xml",
                type="text/xml" # Set correct MIME type
            )

        raise PreventUpdate

    # ----------------------------------------------------------------------
    # 8. Get total portfolio value dynamically to displauy while editing
    # ----------------------------------------------------------------------

    @app.callback(
        Output('total-portfolio-balance', 'children'),
        # 1. Triggers for Reset, Upload, and Add Account (when rowData is set)
        Input('portfolio-grid', 'rowData'), 
        # 2. Triggers for manual user edits to a cell
        Input('portfolio-grid', 'cellValueChanged'),
        prevent_initial_call=True
    )
    def update_total_portfolio_balance(row_data, cell_value_change):
        # When either Input fires, 'row_data' contains the current state of the grid.
        
        # Check if the data is valid before proceeding
        if not row_data:
            # When using multiple Inputs, you should typically use dash.no_update 
            # or raise PreventUpdate if you don't want to change the output.
            # Returning a string works fine if the initial placeholder is set.
            return "NO DATA *** Total Portfolio Balance: $0.00"

        total_balance = 0.0
        for row in row_data:
            try:
                # The balance column in the grid may still contain a string, so clean_currency is essential here.
                balance = clean_currency(row.get('balance', 0.0))
                total_balance += balance
            except (ValueError, TypeError):
                # Safely ignore rows with non-numeric or malformed balances
                pass

        # Use the imported utility to format the output string
        formatted_total = format_currency_output(total_balance, 0) # 0 decimal places

        return f"Total Portfolio Balance: {formatted_total}"
    @callback(
        Output('total-portfolio-balance', 'children'),
        # This input triggers whenever the grid's rowData changes (edits, add, reset, upload)
        Input('portfolio-grid', 'rowData'),
        prevent_initial_call=True
    )
    def update_total_portfolio_balance(row_data):
        if not row_data:
            return "NO DATA *** Total Portfolio Balance: $0.00"

        total_balance = 0.0
        for row in row_data:
            # The 'balance' value must be extracted and cleaned if necessary
            # Assuming the 'balance' field is stored as a float/int:
            try:
                balance = clean_currency(row.get('balance', 0.0))
                total_balance += balance
            except (ValueError, TypeError):
                # Handle cases where the input might not be a clean number yet
                pass

        # You should use a currency formatting utility here if you have one.
        # Otherwise, simple formatting works:
        formatted_total = format_currency_output(total_balance, 0) #second argument is decimal places

        return f"Total Portfolio Balance: {formatted_total}"


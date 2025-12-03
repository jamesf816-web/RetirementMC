Step 11: Install & Run
Still in ~/retirement_calculator:

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt --upgrade
python3 app.py

Open http://127.0.0.1:8050 in your browser.
You'll see a slider to tweak simulation count—drag it,
and watch the success rate bar chart update live
(currently a placeholder at 100%, but it's wired for
your full Monte Carlo).

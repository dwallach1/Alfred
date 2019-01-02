# Alfred

Current Build Status: Compiling, but output is only zeros  

Run `python3 src/main.py`, it will take around 5 minutes to load the models, then it will be ready to handle questions.
It will ask `What is your question?` and then spit out a prediction array for each likelihood array where each index correlates to an existing question in the database.

Example:
> What is your question?
<br>
> Is piped text compatible with Web Services?
<br>
> [0.12, 0.00, ... 0.92, 0.18, 0.02]

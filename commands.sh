## Virtual environnment 
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

## Compute 100(1-α)% PIs for different values of α
python3 get_pred_intervals.py --method=bayes
python3 get_pred_intervals.py --method=conformal

## Evalutate methods
for i in 1 2 3 4 5
do
    python3 analyze_results.py --result_type=$i
done
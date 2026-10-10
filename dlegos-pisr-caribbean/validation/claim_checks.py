import json, math, pathlib, sys

REQUIRED = ["rmse","mae"]

def finite(x):
    return isinstance(x,(int,float)) and math.isfinite(x)

def validate_metrics(path):
    obj=json.loads(pathlib.Path(path).read_text())
    errors=[]
    for model, vals in obj.items():
        for k in REQUIRED:
            if k not in vals or not finite(vals[k]):
                errors.append(f"{model}: missing/nonfinite {k}")
        for k,v in vals.items():
            if isinstance(v,(int,float)) and not math.isfinite(v):
                errors.append(f"{model}: nonfinite {k}")
    return errors

def main():
    if len(sys.argv)<2:
        raise SystemExit("usage: python claim_checks.py metrics.json")
    errors=validate_metrics(sys.argv[1])
    report={"passed":not errors,"errors":errors}
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not errors else 1)

if __name__=="__main__":
    main()

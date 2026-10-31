"""Evaluate released PrecipCorrDiff output against supplied independent truth.
No metrics are generated without a real matching reference dataset.
"""
import argparse, json, pathlib
import numpy as np
import xarray as xr

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prediction",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--variable",default="precipitation")
    p.add_argument("--output",default="precipcorrdiff_metrics.json")
    a=p.parse_args()
    with xr.open_dataset(a.prediction) as pred, xr.open_dataset(a.reference) as ref:
        if a.variable not in pred or a.variable not in ref:
            raise ValueError("Required precipitation variable missing in prediction or reference")
        x,y=xr.align(pred[a.variable],ref[a.variable],join="exact")
        if not x.dims or x.shape!=y.shape: raise ValueError("Prediction/reference mismatch")
        x=x.load().values.astype("float64"); y=y.load().values.astype("float64")
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<10: raise ValueError("Insufficient valid paired observations")
    d=x[good]-y[good]
    out={"n":int(good.sum()),"mae_mm_day":float(np.abs(d).mean()),"rmse_mm_day":float(np.sqrt(np.mean(d*d))),"bias_mm_day":float(d.mean()),"provenance":{"prediction":a.prediction,"reference":a.reference,"variable":a.variable},"note":"Metrics valid only if independent reference provenance and coordinates are verified."}
    pathlib.Path(a.output).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()

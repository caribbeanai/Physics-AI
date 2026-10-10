"""Authenticated, bounded ERA5 retrieval for manually triggered GitHub Actions.

CDSAPI_KEY is read from the process environment; it is never printed or stored.
Default sample: 2017-01-01 through 2017-01-02, 00/06/12/18 UTC.
Monthly mode: all days of one month at the same 6-hour interval.
"""
import argparse
import calendar
import hashlib
import json
import os
from pathlib import Path

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
SHORT_NAMES = {"t2m", "d2m", "sp", "u10", "v10"}
AREA = [30, -90, 5, -55]

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "monthly"], default="sample")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/era5")
    args = parser.parse_args()
    if not (1981 <= args.year <= 2025 and 1 <= args.month <= 12):
        parser.error("year must be 1981–2025 and month 1–12")
    key = os.environ.get("CDSAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("CDSAPI_KEY GitHub Actions secret is missing; configure it in repository Actions secrets.")
    import cdsapi
    import xarray as xr
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"era5_caribbean_{args.year}_{args.month:02d}_{args.mode}"
    target = out / f"{stem}.nc"
    n_days = 2 if args.mode == "sample" else calendar.monthrange(args.year, args.month)[1]
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(args.year)],
        "month": [f"{args.month:02d}"],
        "day": [f"{d:02d}" for d in range(1, n_days + 1)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }
    # No printing of credentials or complete client config.
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=key,
        quiet=True,
        progress=False,
    )
    print(f"Requesting {args.mode} ERA5 Caribbean {args.year}-{args.month:02d} (6-hourly, 5 variables).")
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    if not target.exists() or target.stat().st_size < 1000:
        raise RuntimeError("CDS returned no usable NetCDF file.")
    with xr.open_dataset(target) as ds:
        found = set(ds.data_vars)
        missing = SHORT_NAMES - found
        if missing:
            raise RuntimeError(f"Missing ERA5 variables: {sorted(missing)}; found {sorted(found)}")
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        if time_name not in ds.coords:
            raise RuntimeError("No valid_time/time coordinate in downloaded ERA5 dataset.")
        for coord in ("latitude", "longitude"):
            if coord not in ds.coords:
                raise RuntimeError(f"Missing coordinate: {coord}")
        if ds.sizes[time_name] < 4:
            raise RuntimeError("Too few timestamps in ERA5 response.")
        metadata = {
            "source": "Copernicus Climate Data Store: reanalysis-era5-single-levels",
            "mode": args.mode,
            "year": args.year,
            "month": args.month,
            "area_nwse": AREA,
            "variables": sorted(found),
            "time_coordinate": time_name,
            "time_count": int(ds.sizes[time_name]),
            "latitude_count": int(ds.sizes["latitude"]),
            "longitude_count": int(ds.sizes["longitude"]),
            "size_bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "note": "Data acquisition verification only. No climate model training or scientific benchmark was run.",
        }
    (out / f"{stem}_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("ERA5 retrieval and NetCDF validation succeeded; manifest saved.")
    print(f"Shape: {metadata['time_count']} times × {metadata['latitude_count']} lat × {metadata['longitude_count']} lon")

if __name__ == "__main__":
    run()

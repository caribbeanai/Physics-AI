from pathlib import Path
import cdsapi

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

def download_year(year:int, out_dir:str="data/raw", area=(30,-90,5,-55)):
    """Download ERA5 hourly single-level data for the Caribbean.
    Requires Copernicus CDS credentials in ~/.cdsapirc.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"era5_caribbean_{year}.nc"
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1,13)],
            "day": [f"{d:02d}" for d in range(1,32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": list(area),
        },
        str(target),
    )
    return target

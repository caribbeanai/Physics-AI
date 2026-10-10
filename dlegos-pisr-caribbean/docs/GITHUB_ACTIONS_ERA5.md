# ERA5 via GitHub Actions (CDSAPI_KEY)

The workflow is at `.github/workflows/dlegos-era5.yml` in the repository root.

## One-time setup

1. At https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels, log in and accept the dataset licence.
2. In GitHub, open **Settings → Secrets and variables → Actions → New repository secret**.
3. Create a repository secret named **Climatedatapull** and paste your personal CDS API token as its value. Do **not** put the token in code, a pull request, a chat message, or a workflow input.
4. Open **Actions → D-LEGOS ERA5 Caribbean acquisition → Run workflow** on the `main` branch. Keep defaults (`sample`, `2017`, `1`) for the first test.
5. When successful, download the run artifact `era5-caribbean-sample-2017-1` from the workflow run's **Artifacts** section. It includes the NetCDF data and a JSON provenance manifest.

## Scope and limitations

The default retrieval is January 1–2, 2017 at 00:00, 06:00, 12:00, and 18:00 UTC, for five ERA5 single-level variables over 5°N–30°N and 90°W–55°W. It validates data acquisition and schema only; it does **not** train D-LEGOS or generate climate performance metrics.

Select `monthly` for a whole month at 6-hourly cadence after the sample succeeds. Full 1981–2025 acquisition requires repeated monthly runs and persistent storage beyond GitHub's 7-day artifact retention. Large training jobs will need appropriate compute; GitHub-hosted standard runners are not a substitute for a long GPU training environment.

## Security

The workflow maps the GitHub secret **Climatedatapull** into the step-local environment variable **CDSAPI_KEY**.\n\nThe workflow is **manual-only** (`workflow_dispatch`), not triggered by untrusted pull requests. It has `contents: read` permission and injects `secrets.CDSAPI_KEY` only into the retrieval step. The Python client reads the secret from its environment, never prints or writes it, and does not persist a `.cdsapirc` file. Logs contain the request's public area/time/variable metadata, not the API token.

GitHub secret availability cannot be verified by inspecting workflow YAML. A missing secret fails with a generic error. Keep the token only in GitHub's secret store.

## Troubleshooting

- **CDSAPI_KEY missing**: confirm the GitHub repository secret is named **Climatedatapull** exactly.
- **403 / terms not accepted**: accept ERA5 licence while signed into the same CDS account.
- **CDS queued / timeout**: rerun the bounded sample; large monthly requests can take longer.
- **Wrong NetCDF variable names**: inspect the manifest validation error and adapt the parser only after checking the actual file schema.

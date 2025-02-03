import xarray as xr


def _main():
    ds = xr.open_mfdataset(
        "ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/HALO-*.zarr",
        engine="zarr",
        combine_attrs="drop_conflicts",
    )
    ds.attrs["title"] = "HALO position and attitude data during the PERCUSION campaign"
    ds.attrs["summary"] = (
        "This dataset provides a best estimate of the position and attitude of the HALO aircraft during the ORCESTRA campaign. "
        "In general, 100Hz data from the IGI system is used (see the `source` attribute of the per-flight datasets when in doubt)."
    )

    ds.chunk(time=262144).to_zarr("position_attitude.zarr", mode="w")


if __name__ == "__main__":
    _main()

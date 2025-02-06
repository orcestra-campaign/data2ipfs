import glob
from datetime import datetime
from zoneinfo import ZoneInfo

import fsspec
import numcodecs
import numpy as np
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("SOUNDING",):
            chunks = {
                "SOUNDING": 82,
            }
        case ("SOUNDING", "PRES"):
            chunks = {
                "SOUNDING": 82,
                "PRES": 2240,
            }

    return tuple((chunks[d] for d in dimensions))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": get_chunks(dataset[var].dims),
            "compressor": codec,
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


def open_dataset(filename):
    ds = xr.open_dataset(filename, decode_cf=False)
    ds.FULL_TIME.values -= 712224
    ds = xr.decode_cf(ds)

    ds.TIME.attrs.pop("comment")
    ds.FULL_TIME.attrs.pop("comment")

    ds = ds.swap_dims(
        {"TIME": "SOUNDING", "LATITUDE": "SOUNDING", "LONGITUDE": "SOUNDING"}
    )

    return ds


def main():
    root = "ipns://latest.orcestra-campaign.org"
    datasets = [
        open_dataset(fsspec.open_local(f"simplecache::ipns://{f}"))
        for f in sorted(
            fsspec.filesystem("ipns").glob(
                f"{root}/raw/METEOR/CTD/nc/met_203_1_ctd_*.nc"
            )
        )
    ]
    ds = xr.concat(datasets, dim="SOUNDING", combine_attrs="drop_conflicts")
    ds = ds.assign_coords(SOUNDING=range(1, ds.sizes["SOUNDING"] + 1))
    ds.SOUNDING.attrs = {"long_name": "sounding id", "units": "1"}

    # Re-define global attributes after concatenation
    ds.attrs["featureType"] = "trajectoryProfile"
    ds.attrs = {k: v for k, v in ds.attrs.items() if v != "void"}

    ds.TIME.encoding["units"] = "seconds since 1970-01-01"
    ds.FULL_TIME.encoding["units"] = "seconds since 1970-01-01"
    ds.attrs["time_coverage_start"] = str(ds.TIME.values[0])
    ds.attrs["time_coverage_end"] = str(ds.TIME.values[-1])

    ds.attrs["geospatial_lat_min"] = ds.LATITUDE.min().values
    ds.attrs["geospatial_lat_max"] = ds.LATITUDE.max().values
    ds.attrs["geospatial_lon_min"] = ds.LONGITUDE.min().values
    ds.attrs["geospatial_lon_max"] = ds.LONGITUDE.max().values

    # Set global attributes accoring to ORCESTRA conventions
    ds.attrs["title"] = (
        "GEOMAR PO-processed CTD data of cruise Meteor 203/1 CTD station number 1"
    )

    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["source"] = "CTD profile observation"

    now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["history"] = (
        "created during CTD processing; {now}: converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    )

    ds.attrs["license"] = "CC-BY-4.0"

    encoding = {
        **get_encoding(ds),
        "TIME": {"units": "seconds since 1970-01-01", "dtype": "f8"},
        "FULL_TIME": {"units": "seconds since 1970-01-01", "dtype": "f8"},
    }
    ds.to_zarr("CTD.zarr", encoding=encoding, mode="w", zarr_format=2)


if __name__ == "__main__":
    main()

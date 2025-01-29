from datetime import datetime
from zoneinfo import ZoneInfo

import fsspec
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "range"):
            chunks = {
                "time": 2**10,
                "range": 2**8,
            }
        case ("time", "range_hr"):
            chunks = {
                "time": 2**15,
                "range_hr": 8,
            }
        case ("time", "layer"):
            chunks = {
                "time": 2**17,
                "layer": 3,
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


def main():
    root = "ipns://latest.orcestra-campaign.org"
    ds = xr.open_mfdataset(
        fsspec.open_local(f"simplecache::{root}/raw/METEOR/ceilometer/*/*.nc")
    )

    ds.attrs["featureType"] = "trajectoryProfile"

    ds.attrs["title"] = (
        "Ceilometer (CHM15k Nimbus) measurements during METEOR cruise M203"
    )
    ds.attrs["creator_name"] = "Friedhelm Jansen"
    ds.attrs["creator_email"] = "friedhelm.jansen@mpimet.mpg.de"
    ds.attrs["project"] = "ORCESTRA, BOW-TIE"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["license"] = "CC-BY-4.0"

    now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["history"] = (
        f"{now}: converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    )

    ds.attrs.pop("day")
    ds.attrs.pop("month")
    ds.attrs.pop("year")

    ds.chunk(time=2**18).to_zarr("CHM170158.zarr", encoding=get_encoding(ds), mode="w")


if __name__ == "__main__":
    main()

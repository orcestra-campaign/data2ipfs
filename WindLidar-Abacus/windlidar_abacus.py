from datetime import datetime
from zoneinfo import ZoneInfo

import fsspec
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time", "height"):
            chunks = {
                "time": 2**12,
                "height": 64,
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
    versions = {
        ("v0.0", "Wind LiDAR LiTra S raw data (ship motion present)"),
        ("v1.0", "Wind LiDAR LiTra S heave-corrected, synchronous ship motion data"),
        ("v2.0", "Wind LiDAR LiTra S heave-corrected, asynchronous ship motion data"),
    }

    root = "ipns://latest.orcestra-campaign.org"
    for version, title in versions:
        urlpath = fsspec.open_local(
            f"simplecache::{root}/raw/METEOR/WindLidar-Abacus/{version}/nc_{version}/*.nc"
        )
        ds = xr.open_mfdataset(
            urlpath, chunks={"time": -1}, combine_attrs="drop_conflicts"
        )

        ds.attrs["featureType"] = "trajectoryProfile"

        ds.attrs["title"] = title
        ds.attrs["creator_name"] = "Ilya Serikov"
        ds.attrs["creator_email"] = "ilya.serikov@mpimet.mpg.de"
        ds.attrs["project"] = "ORCESTRA, BOW-TIE"
        ds.attrs["platform"] = "RV METEOR"
        ds.attrs["license"] = "CC-BY-4.0"

        now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
        ds.attrs["history"] = (
            f"{now}: converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
        )

        ds.chunk(time=-1).to_zarr(
            f"{version}.zarr", encoding=get_encoding(ds), mode="w"
        )


if __name__ == "__main__":
    main()

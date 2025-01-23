from datetime import datetime
from zoneinfo import ZoneInfo

import numcodecs
import xarray as xr


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "lim"):
            chunks = {
                "time": 2**18,
                "lim": 2,
            }
        case ("time", "alt"):
            chunks = {
                "time": 2**10,
                "alt": 242,
            }
        case ("time", "alt", "lim"):
            chunks = {
                "time": 2**10,
                "alt": 242,
                "lim": 2,
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
    for dataset in ("LICHT-LIDAR_b", "LICHT-LIDAR_t"):
        ds = xr.open_mfdataset(
            f"ql24??/{dataset}-*.nc", chunks={"time": -1, "alt": -1}
        ).load()
        ds.attrs["start_time"] = str(ds.time.values[0])
        ds.attrs["stop_time"] = str(ds.time.values[-1])

        if dataset.endswith("_b"):
            ds.attrs["title"] = (
                "Raman LiDAR LICHT fast product (2min smoothing) during METEOR cruise M203"
            )
        else:
            ds.attrs["title"] = (
                "Raman LiDAR LICHT slow product (58min smoothing) during METEOR cruise M203"
            )

        ds.attrs["creator_name"] = "Ilya Serikov"
        ds.attrs["creator_email"] = "ilya.serikov@mpimet.mpg.de"
        ds.attrs["project"] = "ORCESTRA, BOW-TIE"
        ds.attrs["platform"] = "RV METEOR"
        ds.attrs["source"] = "Raman Lidar LICHT"
        ds.attrs["license"] = "CC-BY-4.0"

        now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
        ds.attrs["history"] += (
            f"; {now}: converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
        )

        ds.to_zarr(f"{dataset}.zarr", mode="w", encoding=get_encoding(ds))


if __name__ == "__main__":
    main()

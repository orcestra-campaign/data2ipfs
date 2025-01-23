import argparse
import pathlib
import subprocess

import numcodecs
import numpy as np
import xarray as xr

from orcestra.postprocess.level0 import bahamas
from orcestra.io import read_igi, read_bahamas_100hz


_vars = {
    "IRS_LON": ("lon", dict(long_name="WGS84 Datum/Longitude", unit="degrees_east")),
    "IRS_LAT": ("lat", dict(long_name="WGS84 Datum/Latitude", unit="degrees_north")),
    "IRS_ALT": ("alt", dict(long_name="WGS84 Datum/Elliptical Height", unit="m")),
    "IRS_PHI": ("roll", dict(long_name="Attitude/Roll", unit="degree")),
    "IRS_THE": ("pitch", dict(long_name="Attitude/Pitch", unit="degree")),
    "IRS_R": ("heading", dict(long_name="Attitude/Yaw", unit="degree")),
}


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=2)
    delta = numcodecs.Delta("i4")

    return {
        var: {
            "chunks": (2**18,),
            "compressor": codec,
            "filters": [delta],
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


def get_latest(datadir):
    files = list(datadir.iterdir())

    dirname = datadir.name
    y, m, d = dirname[5:9], dirname[9:11], dirname[11:13]
    flight_date = f"{y}-{m}-{d}"

    for f in files:
        if "100Hz" in f.name:
            if flight_date == "2024-11-10":
                ds = read_igi(f, skip_header=99, flight_date=flight_date)
                ds = ds.drop_vars(
                    ["IGI_RMSX", "IGI_RMSY", "IGI_RMSZ", "IRS_EWV", "IRS_NSV", "IRS_VV"]
                )
            else:
                ds = read_bahamas_100hz(f, flight_date=flight_date)
            ds.attrs["source"] = "BAHAMAS IGI 100 Hz"

            return ds

    for f in files:
        if "10Hz" in f.name:
            ds = read_igi(f, flight_date=flight_date)[_vars.keys()]
            ds.attrs["source"] = "BAHAMAS IGI 10 Hz"

            return ds

    for f in files:
        if f.suffix == ".nc":
            ds = xr.open_dataset(f).pipe(bahamas)[_vars.keys()]
            ds.attrs["source"] = "BAHAMAS Quick look"

            return ds


def apply_scale(dataset):
    scale_factors = {
        "lat": 1_000_000,
        "lon": 1_000_000,
        "alt": 100_000,
        "roll": 100_000,
        "pitch": 100_000,
        "heading": 100_000,
    }

    return dataset.assign(
        {
            v: xr.DataArray(
                (dataset[v] * s).astype(np.int32),
                attrs={
                    **dataset[v].attrs,
                    "scale_factor": 1 / s,
                },
            )
            for v, s in scale_factors.items()
        }
    )


def homogenize(ds):
    for old_name, (new_name, attrs) in _vars.items():
        ds = ds.assign({new_name: ds[old_name].assign_attrs(attrs)}).drop_vars(old_name)

    ds = ds.assign_coords(lat=ds.lat, lon=ds.lon, alt=ds.alt).pipe(apply_scale)
    ds.attrs["version"] = "1.0.0"

    ds.attrs["title"] = "HALO position and attitude data"
    ds.attrs["summary"] = (
        "This dataset provides a best estimate of the position and attitude of the HALO aircraft during the ORCESTRA campaign. The data is collected from the IGI system (see `source` attribute)."
    )
    ds.attrs["creator_name"] = "Lukas Kluft"
    ds.attrs["creator_email"] = "lukas.kluft@mpimet.mpg.de"
    ds.attrs["project"] = "ORCESTRA, PERCUSION"
    ds.attrs["platform"] = "HALO"
    ds.attrs["license"] = "CC0-1.0"

    return ds


def _halo20240827_hack(raw, products):
    """HALO-20240827a/b hack"""
    ds_a = read_igi(
        raw / "HALO-20240827a/QL_HALO-20240827a_IGI_10Hz_V01.txt", "2024-08-27"
    )
    ds_a = ds_a[_vars.keys()].pipe(homogenize)

    ds_b = xr.open_dataset(raw / "HALO-20240827a/QL_HALO-20240827b_BAHAMAS_V01.nc")
    ds_b = ds_b.pipe(bahamas)[_vars.keys()].pipe(homogenize)

    ds = xr.concat([ds_a, ds_b], dim="time")
    ds.attrs["source"] = (
        "Here be dragons! During the 2024-08-27 flight, the BAHAMAS system failed. As a result, position and attitude data are being combined from two separate input sources with a small gap between them."
    )

    store = products / "HALO-20240827a.zarr/"
    ds.to_zarr(
        store,
        encoding=get_encoding(ds),
        mode="w",
        zarr_format=2,
    )


def _main():
    parser = argparse.ArgumentParser(prog="bahamas2ipfs")
    parser.add_argument(
        "--products",
        "-p",
        type=pathlib.Path,
        default="./",
    )
    parser.add_argument(
        "--raw",
        "-r",
        type=pathlib.Path,
        default="/work/mh0010/ORCESTRA/raw/HALO/bahamas/",
    )

    args = parser.parse_args()

    for flight in sorted(args.raw.glob("HALO-*")):
        if flight.name == "HALO-20240827a":
            _halo20240827_hack(args.raw, args.products)
        else:
            store = args.products / flight.with_suffix(".zarr").name
            ds = get_latest(flight).pipe(homogenize)
            ds.to_zarr(
                store,
                encoding=get_encoding(ds),
                mode="w",
                zarr_format=2,
            )


if __name__ == "__main__":
    _main()

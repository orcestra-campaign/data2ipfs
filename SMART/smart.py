import pathlib

import numcodecs
import numpy as np
import xarray as xr


def get_chunks(sizes):
    match tuple(sizes.keys()):
        case ("wavelength", "time"):
            chunks = {
                "time": 2**16,
                "wavelength": 7,
            }
        case (single_dim,):
            chunks = {
                single_dim: sizes[single_dim],
            }
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    compressor = numcodecs.Blosc("zstd", clevel=6)

    return {
        var: {
            "compressor": compressor,
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
    }


def fix_time(ds, gps_time_offset=np.timedelta64(18, "s")):
    fid = pathlib.Path(ds.encoding["source"]).name
    y, m, d = fid[54:58], fid[58:60], fid[60:62]

    seconds = ds.time.values
    # new_day = np.where(np.diff(milliseconds) < -86_399_999)[0]
    # if new_day.size > 0:
    #    # This correcrts date changes during HALO flights where
    #    # the internal "milliseconds since midnight" time is reset.
    #    milliseconds[int(new_day[0]) + 1 :] += 86_400_000

    time = np.datetime64(f"{y}-{m}-{d}") + seconds * np.timedelta64(1, "s")
    time = time.astype("datetime64[ns]")  # Xarray compatibility

    return ds.assign_coords(time=time)


def main():
    attrs_per_var = {
        "Iup_meas": {
            "units": "W m-2 nm-1 sr-1",
            "long_name": "Calibrated upward Radiance",
        },
        "Fup_meas": {
            "units": "W m-2 nm-1",
            "long_name": "Calibrated upward Irradiance",
        },
        "Fdw_sim": {
            "units": "W m-2 nm-1",
            "long_name": "Simulated downward Irradiance",
        },
    }

    for direction in ("Fup", "Iup"):
        ds = xr.open_mfdataset(
            f"*_{direction}.nc",
            preprocess=fix_time,
            concat_dim="time",
            combine="nested",
        )

        wavelength = xr.DataArray(
            data=np.array(sorted([442, 559, 832, 945, 1373, 1613, 1100])),
            dims=("wavelength",),
            name="wavelength",
            attrs={"units": "nm"},
        )
        ds = ds.assign_coords(wavelength=wavelength)

        for var_prefix in (f"{direction}_meas", "Fdw_sim"):
            arr = [ds[f"{var_prefix}_{w}_nm"].values for w in ds.wavelength.values]
            ds = ds.assign(
                {
                    var_prefix: (
                        ("wavelength", "time"),
                        np.stack(arr),
                        attrs_per_var[var_prefix],
                    )
                }
            ).drop_vars([f"{var_prefix}_{w}_nm" for w in ds.wavelength.values])

        ds.load().to_zarr(
            f"SMART_{direction}.zarr",
            encoding=get_encoding(ds),
            zarr_format=2,
            mode="w",
        )


if __name__ == "__main__":
    main()

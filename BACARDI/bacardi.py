import glob

import numcodecs
import numpy as np
import xarray as xr


def fix_time(ds, gps_time_offset=np.timedelta64(18, "s")):
    fid = ds.attrs["flightname"]
    y, m, d = fid[5:9], fid[9:11], fid[11:13]

    milliseconds = ds.TIME.values
    new_day = np.where(np.diff(milliseconds) < -86_399_999)[0]
    if new_day.size > 0:
        # This correcrts date changes during HALO flights where
        # the internal "milliseconds since midnight" time is reset.
        milliseconds[int(new_day[0]) + 1 :] += 86_400_000

    time = np.datetime64(f"{y}-{m}-{d}") + milliseconds * np.timedelta64(1, "ms")
    time = time.astype("datetime64[ns]")  # Xarray compatibility

    return ds.swap_dims(tid="TIME").assign_coords(TIME=time)


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": (2**17,),
            "compressor": codec,
        }
        if var not in dataset.dims
        else {"chunks": -1}
        for var in dataset.variables
    }


def main():
    ds = xr.open_mfdataset(
        sorted(glob.glob("*.nc")),
        concat_dim="TIME",
        combine="nested",
        combine_attrs="drop_conflicts",
        preprocess=fix_time,
    ).sortby("TIME")

    for var in ds.variables:
        if "units" in ds[var].attrs:
            ds[var].attrs["units"] = (
                ds[var]
                .attrs["units"]
                .replace("W/m^2", "W m-2")
                .replace("deg/s", "deg s-1")
            )

    ds.IRS_LAT.attrs["units"] = "degrees_north"
    ds.IRS_LON.attrs["units"] = "degrees_east"
    ds.attrs["featureType"] = "trajectory"

    ds.attrs["title"] = "Broadband solar and terrestrial, upward and downward irradiance measured by BACARDI on HALO during the PERCUSION field campaign"
    ds.attrs["keywords"] = "airborne measurements, broadband irradiance, irradiance, solar irradiance, terrestrial irradiance, radiometer, aircraft"

    ds.attrs["creator_name"] = (
        "A. Giez, M. Zoeger, Ch. Mallaun, V. Nenakhov, L. Eirenschmalz, D. Pasternak"
    )
    ds.attrs["creator_email"] = "andreas.giez@dlr.de"

    ds.attrs["history"] = "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds.attrs["references"] = "https://halo-db.pa.op.dlr.de/dataset/10435"
    ds.attrs["license"] = "CC-BY-4.0"

    ds.chunk(TIME=-1).to_zarr(
        "BACARDI.zarr", encoding=get_encoding(ds), mode="w", zarr_format=2
    )


if __name__ == "__main__":
    main()

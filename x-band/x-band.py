import numcodecs
import xarray as xr


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": (2**18,),
            "compressor": codec,
        }
        if var not in dataset.dims
        else {"chunks": dataset[var].size}
        for var in dataset.variables
    }


def main():
    ds = xr.open_dataset(
        "Radar_Derived_Currents_M203_20240816T193520_to_20240923T163655.nc"
    ).rename_dims(n_measurment="measurement")
    ds.attrs["featureType"] = "trajectory"

    # Set global attributes accoring to ORCESTRA conventions
    ds.attrs["summary"] = ds.attrs.pop("comment")
    ds.attrs["creator_name"] = "Jan Boedewadt, Ruben Carrasco, Jochen Horstmann"
    ds.attrs["creator_email"] = ", ruben.carrasco@hereon.de, "
    ds.attrs["history"] = (
        f"Original dataset created by {ds.originator} ({ds.contact}) on {ds.creation_date}; Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    )
    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["platform"] = "RV METEOR"
    ds.attrs["project"] = "ORCESTRA, BOW-TIE"

    ds.chunk(measurement=-1).to_zarr(
        "Radar_Derived_Currents_M203.zarr",
        encoding=get_encoding(ds),
        mode="w",
        zarr_format=2,
    )


if __name__ == "__main__":
    main()

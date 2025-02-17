import fsspec
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
        else {"chunks": -1}
        for var in dataset.variables
    }


def main():
    root = "QmbWMPVWkqBCKZexe9HrvCztCyPUPyPRSsCPpAUxEnovj6"
    datasets = [
        xr.open_dataset(
            fsspec.open_local(f"simplecache::ipfs://{f}"), chunks={"tid": -1}
        )
        for f in sorted(fsspec.filesystem("ipfs").glob(f"{root}/HALO-2024?????/*.nc"))
    ]
    ds = (
        xr.concat(datasets, dim="tid", combine_attrs="drop_conflicts")
        .swap_dims(tid="TIME")
        .sortby("TIME")
    )
    ds.attrs["featureType"] = "trajectory"

    ds.IRS_LAT.attrs["units"] = "degrees_north"
    ds.IRS_LON.attrs["units"] = "degrees_east"

    # Set global attributes accoring to ORCESTRA conventions
    ds.attrs["title"] = "BAHAMAS QL measurements"
    ds.attrs["summary"] = "Basic meteorological and avionic data of HALO."
    ds.attrs["project"] = "ORCESTRA, PERCUSION"

    ds.attrs["creator_name"] = (
        "Christian Mallaun, Dominika Pasternak, Lisa Eirenschmalz"
    )
    ds.attrs["creator_email"] = (
        "Christian.Mallaun@DLR.DE, Dominika.Pasternak@DLR.DE, Lisa.Eirenschmalz@DLR.DE"
    )

    ds.attrs["history"] = "converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds.attrs["license"] = "CC-BY-4.0"

    ds.chunk(TIME=-1).to_zarr(
        "BAHAMAS_QL.zarr", encoding=get_encoding(ds), mode="w", zarr_format=2
    )


if __name__ == "__main__":
    main()

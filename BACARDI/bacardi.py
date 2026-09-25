import glob
import json

import numcodecs
import xarray as xr


def get_creator_name():
    with open("18999497.json", "r") as fp:
        meta = json.loads(fp.read())

    creator_names = []
    for creator in meta["metadata"]["creators"]:
        person = creator["person_or_org"]
        family_name = person["family_name"]
        given_name = person["given_name"]

        creator_names.append(f"{given_name} {family_name}")

    return ",".join(creator_names)


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": (2**17,),
            "compressor": codec,
        }
        for var in dataset.variables
    }


def main():
    ds = xr.open_mfdataset(
        sorted(glob.glob("HALO-DB_dataset*_release2_*_BACARDI_BroadbandFluxes")),
        concat_dim="time",
        combine="nested",
        combine_attrs="drop_conflicts",
        engine="netcdf4",
    ).sortby("time")

    ds.attrs["featureType"] = "trajectory"

    ds.attrs["title"] = (
        "Broadband solar and thermal-infrared, upward and downward irradiance measured by BACARDI during the PERCUSION field campaign"
    )
    ds.attrs["keywords"] = (
        "airborne measurements, broadband irradiance, irradiance, solar irradiance, terrestrial irradiance, radiometer, aircraft, PERCUSION"
    )
    ds.attrs["summary"] = open("summary.md", "r").read()

    ds.attrs["creator_name"] = get_creator_name()

    ds.attrs["history"] = "Converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
    ds.attrs["references"] = ",".join(
        sorted(
            {
                "https://doi.org/10.5281/zenodo.18999497",
                "https://doi.org/10.5194/amt-16-1563-2023",
            }
        )
    )
    ds.attrs["license"] = "CC-BY-4.0"

    ds.chunk(time=-1).to_zarr(
        "BACARDI.zarr", encoding=get_encoding(ds), mode="w", zarr_format=2
    )


if __name__ == "__main__":
    main()

from datetime import datetime
from zoneinfo import ZoneInfo

import fsspec
import numcodecs
import xarray as xr


def get_chunks(dimensions):
    if "DEPTH" in dimensions:
        chunks = {
            "DEPTH": 4,
            "TIME": 2**16,
        }
    else:
        chunks = {
            "TIME": 2**16,
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
    ncfiles = (
        "met_203_vmadcp_38khz.nc",
        "met_203_vmadcp_75khz.nc",
    )

    root = "ipns://latest.orcestra-campaign.org"
    for ncfile in ncfiles:
        ds = xr.open_dataset(
            fsspec.open_local(
                f"simplecache::{root}/raw/METEOR/ADCP/{ncfile}", chunks={}
            )
        )

        ds.attrs["creator_name"] = "Daniel Klocke, Marcus Dengler, Robert Kopte"
        ds.attrs["creator_email"] = (
            "daniel.klocke@mpimet.mpg.de, mdengler@geomar.de, robert.kopte@ifg.uni-kiel.de"
        )
        ds.attrs["project"] = "ORCESTRA, BOW-TIE, DAM_Underway"
        ds.attrs["instrument"] = ds.attrs["sensor"]
        ds.attrs["platform"] = "RV METEOR"

        ds.attrs["source"] = "ADCP measurements"
        ds.attrs["license"] = "CC-BY-4.0"

        if "38khz" in ncfile:
            ds.attrs["title"] = (
                "Shipboard ADCP current measurements (38 kHz) during RV METEOR cruise M203"
            )
            ds.attrs["references"] = "https://doi.org/10.1594/PANGAEA.974251"
            ds.attrs["summary"] = (
                "Current velocities of the upper water column along the cruise track of R/V Meteor cruise M203 were collected by a vessel-mounted 38 kHz RDI Ocean Surveyor ADCP. The ADCP transducer was located at 5.0 m below the water line. The instrument was operated in narrowband mode (WM10) with a bin size of 32.00 m, a blanking distance of 16.00 m, and a total of 55 bins, covering the depth range between 53.0 m and 1781.0 m. Attitude data from the ship's motion reference unit were used by the data acquisition software VmDAS internally to convert ADCP beam velocities to geographic coordinates. The Python toolbox OSADCP (version 2.0.0) was used for data post-processing. Single-ping data were screened for bottom signals and, where appropriate, a bottom mask was manually processed. Acoustic Interferences were identified based on outliers in the ADCP echo intensity data. Echo intensity data were cleaned accordingly and affected velocity cells were flagged to be removed prior ensemble-averaging. The ship's velocity was calculated from position fixes obtained by the Global Navigation Satellite System (GNSS), taking into account lever arms of ADCP transducer and GNSS antenna. Accuracy of the derived water velocities mainly depends on the quality of the position fixes and the ship's heading data. Further errors stem from a misalignment of the transducer with the ship's centerline. Data processing included water track calibration of the misalignment angle (0.0691° +/- 0.5613°) and scale factor (1.0054 +/- 0.0088) of the measured velocities. The velocity data were averaged in time using an average interval of 60 s. Depth cells with ensemble-averaged percent-good values below 25% are marked as 'bad data'."
            )
        elif "75khz" in ncfile:
            ds.attrs["title"] = (
                "Shipboard ADCP current measurements (75 kHz) during RV METEOR cruise M203"
            )
            ds.attrs["references"] = "https://doi.org/10.1594/PANGAEA.974252"
            ds.attrs["summary"] = (
                "Current velocities of the upper water column along the cruise track of R/V Meteor cruise M203 were collected by a vessel-mounted 75 kHz RDI Ocean Surveyor ADCP. The ADCP transducer was located at 5.0 m below the water line. The instrument was operated in narrowband mode (WM10) with a bin size of 8.00 m, a blanking distance of 4.00 m, and a total of 100 bins, covering the depth range between 17.0 m and 809.0 m. Attitude data from the ship's motion reference unit were used by the data acquisition software VmDAS internally to convert ADCP beam velocities to geographic coordinates. The Python toolbox OSADCP (version 2.0.0) was used for data post-processing. Single-ping data were screened for bottom signals and, where appropriate, a bottom mask was manually processed. Acoustic Interferences were identified based on outliers in the ADCP echo intensity data. Echo intensity data were cleaned accordingly and affected velocity cells were flagged to be removed prior ensemble-averaging. The ship's velocity was calculated from position fixes obtained by the Global Navigation Satellite System (GNSS), taking into account lever arms of ADCP transducer and GNSS antenna. Accuracy of the derived water velocities mainly depends on the quality of the position fixes and the ship's heading data. Further errors stem from a misalignment of the transducer with the ship's centerline. Data processing included water track calibration of the misalignment angle (-46.4281° +/- 0.5612°) and scale factor (1.0069 +/- 0.0086) of the measured velocities. The velocity data were averaged in time using an average interval of 60 s. Depth cells with ensemble-averaged percent-good values below 25% are marked as 'bad data'."
            )

        now = datetime.now().astimezone(ZoneInfo("UTC")).strftime(r"%Y-%m-%dT%H:%M:%SZ")
        ds.attrs["history"] += (
            f"; {now}: converted to Zarr by Lukas Kluft (lukas.kluft@mpimet.mpg.de)"
        )

        ds.to_zarr(
            ncfile.replace(".nc", ".zarr"),
            mode="w",
            encoding=get_encoding(ds),
            zarr_format=2,
        )


if __name__ == "__main__":
    main()

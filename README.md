# Prepare data for publication via IPFS

The official ORCESTRA dataset is distributed via [IPFS](https://ipfs.tech).
The latest version is always available via the IPNS name:

    latest.orcestra-campaign.org

Technically, IPFS is not restricted to any particular data format.
However, certain data formats allow the user to access data directly via http(s).
One of these data formats is the [Zarr] (https://zarr.dev) format, which is designed to store large, chunked, compressed, N-dimensional arrays.

This repository provides Python scripts that convert the various raw data sets collected during ORCESTRA to Zarr.

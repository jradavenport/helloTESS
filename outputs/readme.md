# Outputs

Here we have the output .csv files from the latest run of my simple flare + rotation detection code. Each Sector takes ~1 day to run (on a standard single CPU core). These should be useful as first-pass catalogs for the rotation of stars in TESS 2-min data, and identify many possible flare stars.

A good sample of rotation periods could be found by selecting sources where the Lomb-Scargle and ACF, or BLS and ACF generally agree.

The flare detection is quite simplified, and has both spurious events and many real flares missing. However most large amplitude flares *are* recovered, and should be useful for a first-pass flare analysis.

# Meeting 1 (2021-01-20)
* Improve data loading, some bugs discovered
* Evaluate if separate python script instead of SPM
* Limitations of matlab, one file at a time, can it handle multiple files?
    - with supplied script, yes

## Questions:
* Some folders contain 6 nii files. How to treat:
    - Keep all
    - Select 1
    - Average

* CNN input:
    - Should we mosaic one image as input or take each slice as an input?
    - Each slice as one channel (replacement instead of color)?
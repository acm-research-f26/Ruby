The 79 real lenses in the Ruby test set: 4 compound lens candidates and 75 ordinary lenses.
These are the exact images the models were tested on (the rows with split == test and label == 1 in data/split.csv).

test_lenses.csv  one row per lens: Euclid catalog ID, name (compound lenses only), compound yes/no, expert grade (A or B),
                 RA and Dec in degrees, Euclid's Einstein radius in arcsec (blank where Euclid has no lens model),
                 and the file names below
npy/             raw Euclid VIS cutouts, 100 x 100 pixels at 0.1"/pixel (10" across), float32, no scaling.
                 Load with numpy.load(). Row 0 is the bottom of the image.
png/             the same images for viewing: arcsinh stretch, 4x enlarged, bottom row at the bottom
sheet.png        all 79 on one page

Sources: lens IDs, grades and coordinates are from the Euclid Q1 Strong Lensing Discovery Engine catalog,
doi:10.5281/zenodo.15003116. Cutouts were made from Euclid Q1 VIS images on NASA/IPAC IRSA with make_cutouts.py.
The 4 compound lenses are candidates (Li et al. 2026, arXiv:2503.15327), not spectroscopically confirmed.

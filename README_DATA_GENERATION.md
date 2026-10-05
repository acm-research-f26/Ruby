# How the Sample Images Are Made

This guide explains how `generate_lenses.py` makes the sample images. They are simulated with `lenstronomy`, not taken from telescope surveys, and they are not confirmed real lenses.

## What's In the Dataset

By default, the script creates 2,000 images:

- 500 simple lenses, each with one singular isothermal ellipsoid (SIE) deflector.
- 500 compound lenses, each with two SIE deflectors and two foreground galaxy light profiles.
- 1,000 non-lenses, each with a foreground galaxy and a nearby companion, with almost no lensing.

The notebook originally models lenses with one SIE. I added the two-SIE version in the generator for the compound examples. The non-lenses are based on the notebook's harder companion-galaxy example.

## How the Lenses Are Made

Each lens image has one background galaxy with a Sérsic-ellipse light profile. Its redshift is randomly chosen between 0.3 and 5.0. The source's brightness and size change with redshift:

- Magnitude: `20.0 + 1.5 * redshift`
- Sérsic radius: `0.3 / (1 + 0.5 * redshift)`

The background galaxy is placed slightly off-center. Its x and y offsets are each randomly chosen between 0.05 and 0.2 arcseconds.

For a simple lens, the SIE is centered at (0, 0), and its Einstein radius is randomly chosen between 1.0 and 2.0 arcseconds. For a compound lens, two SIEs sit on opposite sides of the image center, each with an Einstein radius between 0.65 and 1.15 arcseconds. Each one also has its own foreground light profile.

## How the Non-Lenses Are Made

Each non-lens has a foreground galaxy and a nearby companion. The companion's brightness, size, shape, and position are randomized. The SIE Einstein radius is only 0.01 arcseconds, so it has almost no lensing effect. This makes the non-lenses a tougher comparison than images with just one isolated galaxy.

## Image Settings

All images are 64-by-64 pixel grayscale simulations. They use the same Euclid-like single-band settings as the notebook's hard non-lens example:

- Pixel scale: 0.1 arcseconds per pixel
- Exposure time: 90 seconds
- Magnitude zero point: 27.0
- Read noise: 7
- CCD gain: 6.083
- Sky brightness: 23.5
- Seeing: 0.16 arcseconds
- Gaussian PSF, one exposure

`lenstronomy` renders the galaxy light and adds background and Poisson noise. The script clips extreme pixel values, scales the rest to 0-255, and saves each image as a PNG.

## Run It Again

The script uses fixed random seeds, so running it again with the same code and packages makes the same images and CSV values.

From the project folder, install the required packages and run the script:

```powershell
python -m pip install -r requirements.txt
python generate_lenses.py
```

By default, the script writes `synthetic_lenses.csv` and puts the images in `synthetic_lenses_images/`. Simple lenses, compound lenses, and non-lenses each have their own folder. Each CSV row includes the image path and, for lenses, the source redshift. Non-lens rows have a blank redshift.

To save the CSV somewhere else, pass an output path:

```powershell
python generate_lenses.py --output output/my_dataset.csv
```

The image folder is created next to the CSV and uses the same filename with `_images` added.

## A Few Things to Keep in Mind

These are simulated examples for testing the data and model pipeline. They are not telescope observations, and the Euclid-like settings have not been calibrated against Euclid data. The simple and compound labels describe how the generator builds each image; they are not labels from a catalog of confirmed lenses.

The CSV columns `feature_1` through `feature_6` are random placeholders. They are not measured from the images, so don't treat them as scientific features. For image-based training, use the PNG files listed in the CSV's `image_path` column.

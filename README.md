 
# Project Overview & Problem Statement

Project RUBY leverages Deep Learning and Computer Vision techniques to detect rare and exotic Double-Source-Plane Lenses (DSPLs), also known as compound gravitational lenses. These types of lenses occur when multiple background sources at different redshifts align behind a foreground lens galaxy, producing nested Einstein rings at distinct radii.

Large imaging surveys such as Euclid and the Vera Rubin Observatory Legacy Survey of Space and Time are expected to send back ~1 billion images of the universe, and out of these images we expect to find more than 100,000 strong gravitational lens systems, including many rare and exotic populations like the compound lenses mentioned before.

Although various ML algorithms exist for classifying images as lenses vs. non lenses, even state-of-the art algorithms, are not able to classify certain types of compound lenses with
- smaller Einstein radii, 
- fainter arcs 
- lens light blending with the arc light.
Although only candidate DSPLs, these are still systems that we are interested in.
 
Project Ruby utilizes a Zoobot ConvNeXt model combined with model interpretability techniques (such as DeepDream) to understand why state-of-the-art models fail on these edge cases, enabling us to design an optimized architecture tailored for compound lens detection.

# Dataset & Simulation Pipeline
Because real training data for compound lenses is extremely scarce, we simulate realistic astronomical images using Lenstronomy, an open-source Python package for gravitational lens modeling.

## Simulation Specifications:
- Image Dimensions: 100 x 100 pixels
- centered on the foreground lens system.
- Filter Band: Grayscale images using the single IE band (found to perform better than multi-band color images for the model we are expanding upon).
- Survey Parameter Emulation: Parameters are tuned to mirror the point-spread function (PSF) and sensor characteristics of the Euclid survey.

So far, we’ve mainly focused on standard, single-lens systems. This has helped us get familiar with Lenstronomy and establish a baseline before moving on to more complex cases.

## Current Progress & Next Steps in Simulation
Completed: 
Generated standard, single-lens baseline systems to establish pipelines and validate parameter distributions.

## In Progress:
Simulating complex/irregular non-Sérsic background sources. Creating challenging compound lenses with faint, small-radius arcs.

# Model Architecture & Preliminary Results
Model Choice: Zoobot Bayesian CNN
- Part of the Galaxy Zoo citizen Science project, it is pretrained on 92 million morphological classifications of galaxies
- Transfer Learning Approach: Following the European Space Agency (ESA) methodology, early layers are frozen while the final classification head and leading ConvNeXt blocks are fine-tuned.
- Validation: Zoobot was selected because it outperformed 20 alternative machine learning models (including Vision Transformers / ViTs) on Euclid Early Release Observations (ERO)
## Key Experimental Findings So Far
- Simulation-to-Real Domain Gap: Training Zoobot purely on simulated images and testing on real telescope data yielded accuracy only slightly above random guessing (~50%). This matches findings published by the Euclid consortium.
- Real-to-Real Benchmark: Fine-tuning and testing directly on real images produced high accuracy, proving the model structure is sound, but highlighting the need for higher-fidelity simulations.
## Interpretability: 
Looked into utilizing DeepDream + Zoobot to understand what channels are influencing decision making, but inconclusive results. 

# Astrophysical Significance & Future Applications
Hubble Constant: High-precision detection of lensed systems with time-delay measurements aims to constrain the universe's expansion rate <1% uncertainty.
Dark Matter & Dark Energy: Because a single compound lens provides two distinct lens-plane geometries, as few as 50 confirmed compound lenses can constrain the matter density(Gavazzi et al. 2008).Helps map dark matter halos and refine models ranging from the early Big Bang to cosmic expansion.

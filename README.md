# Ruby

Gravitational lensing is the bending of light around high-mass objects so that they behave like a lens. Traditionally, CNN models that detect these systems are biased against high-redshift (high-z) gravitational lenses. This happens because real-world data (such as data from the Sloan Lens ACS survey) is often skewed, leaving models without a clear understanding of what high-z systems look like. Improving high-z lens detection directly expands the sample of lenses available for time-delay cosmography, which can be used to derive the Hubble constant. The aim of this project is to create a CNN-based architecture that performs well across a broader range of high-redshift galaxy-galaxy lens detections.

Past attempts at finding high-redshift galaxy-galaxy lenses failed to address this architectural bias by exclusively training on high-redshift datasets. Ruby addresses this bias at the architectural level by modifying existing CNN galaxy lens detectors designed for low redshift ranges and combining them with a multimodal model built for finding high-redshift lenses. Ruby also incorporates physics-based augmentations, such as redshift-dependent g, r, and i-band flux-shifting, to increase the model's sensitivity to rarer systems. With an influx of data from new space surveys like Euclid and the Legacy Survey of Space and Time (LSST), having a robust automated lens detection system is imperative now more than ever. Ruby lays the foundation for stronger galaxy-to-galaxy lens detection, helping researchers around the world gain a clearer picture of the universe. 




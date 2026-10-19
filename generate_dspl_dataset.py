import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel

output_dir = "dspl_dataset"
os.makedirs(output_dir, exist_ok=True)

# Scale up to 600 total images
num_images = 600  
num_pix = 64
delta_pix = 0.04
transform_pix2coord = np.array([[delta_pix, 0], [0, delta_pix]])
ra_at_xy_0 = -delta_pix * (num_pix - 1) / 2
dec_at_xy_0 = -delta_pix * (num_pix - 1) / 2

pixel_grid = PixelGrid(num_pix, num_pix, transform_pix2coord, ra_at_xy_0, dec_at_xy_0)
psf = PSF(psf_type='GAUSSIAN', fwhm=0.15)

lens_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
source_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
source_model = LightModel(light_model_list=source_model_list)

lens_model_dspl = LensModel(lens_model_list=['SIE', 'SIE'], multi_plane=True, lens_redshift_list=[0.3, 0.6], z_source=1.2)
lens_model_empty = LensModel(lens_model_list=[])

dataset_records = []
print(f"Generating {num_images} randomized images with an automatic train/test split...")

# Create a randomized array to shuffle the 50/50 Lens vs Galaxy distribution evenly
class_labels = np.array([1] * (num_images // 2) + [0] * (num_images // 2))
np.random.shuffle(class_labels)

for i, is_lens in enumerate(class_labels):
    # Determine split subset: 70% Train, 30% Test
    split_group = "train" if np.random.uniform(0, 1) < 0.7 else "test"
    
    amp_lens = np.random.uniform(65000, 105000)
    amp_src1 = np.random.uniform(15000, 55000)
    amp_src2 = np.random.uniform(10000, 35000)
    
    theta_E_main = np.random.uniform(0.5, 0.85)
    theta_E_inter = np.random.uniform(0.1, 0.3)
    
    e1_l = np.random.uniform(-0.15, 0.15)
    e2_l = np.random.uniform(-0.15, 0.15)
    
    is_blended = np.random.choice([0, 1], p=[0.5, 0.5])
    
    kwargs_lens_light = [{
        'amp': amp_lens, 'R_sersic': np.random.uniform(0.09, 0.15), 'n_sersic': np.random.uniform(3.5, 4.5),
        'e1': e1_l, 'e2': e2_l, 'center_x': np.random.uniform(-0.01, 0.01), 'center_y': np.random.uniform(-0.01, 0.01)
    }]
    
    if is_blended == 1:
        kwargs_lens_light.append({
            'amp': np.random.uniform(30000, 60000), 'R_sersic': np.random.uniform(0.07, 0.13), 'n_sersic': 1.0,
            'e1': np.random.uniform(-0.1, 0.1), 'e2': np.random.uniform(-0.1, 0.1), 
            'center_x': np.random.choice([np.random.uniform(-0.9, -0.4), np.random.uniform(0.4, 0.9)]),
            'center_y': np.random.choice([np.random.uniform(-0.9, -0.4), np.random.uniform(0.4, 0.9)])
        })
        extended_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'])
    
    if is_lens == 1:
        kwargs_lens = [
            {'theta_E': theta_E_main, 'e1': e1_l, 'e2': e2_l, 'center_x': 0.0, 'center_y': 0.0},
            {'theta_E': theta_E_inter, 'e1': np.random.uniform(-0.2, 0.2), 'e2': np.random.uniform(-0.2, 0.2), 'center_x': np.random.uniform(-0.05, 0.05), 'center_y': np.random.uniform(-0.05, 0.05)}
        ]
        kwargs_source = [
            {'amp': amp_src1, 'R_sersic': np.random.uniform(0.04, 0.06), 'n_sersic': 1.0, 'e1': np.random.uniform(-0.1, 0.1), 'e2': np.random.uniform(-0.1, 0.1), 'center_x': np.random.uniform(-0.05, 0.05), 'center_y': np.random.uniform(-0.05, 0.05)},
            {'amp': amp_src2, 'R_sersic': np.random.uniform(0.05, 0.08), 'n_sersic': 1.0, 'e1': np.random.uniform(-0.1, 0.1), 'e2': np.random.uniform(-0.1, 0.1), 'center_x': np.random.uniform(-0.08, 0.08), 'center_y': np.random.uniform(-0.08, 0.08)}
        ]
        engine = ImageModel(pixel_grid, psf, lens_model_dspl, source_model, extended_light_model if is_blended == 1 else lens_light_model)
        clean_image = engine.image(kwargs_lens, kwargs_source, kwargs_lens_light)
        proxy_magnitude = 25.0 - 2.5 * np.log10(amp_src2)
        calculated_size = theta_E_main * 2.0
    else:
        kwargs_source_empty = [{'amp': 0, 'R_sersic': 0.1, 'n_sersic': 1.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0, 'center_y': 0}] * 2
        engine = ImageModel(pixel_grid, psf, lens_model_empty, source_model, extended_light_model if is_blended == 1 else lens_light_model)
        clean_image = engine.image(kwargs_lens=[], kwargs_source=kwargs_source_empty, kwargs_lens_light=kwargs_lens_light)
        proxy_magnitude = 0.0 
        calculated_size = 0.0

    background_rms = np.random.uniform(11.0, 19.0)
    noise = np.random.normal(100.0, background_rms, clean_image.shape)
    noisy_image = clean_image + noise
    
    prefix = "lens" if is_lens == 1 else "galaxy"
    npy_filename = os.path.join(output_dir, f"{prefix}_{i:04d}.npy")
    png_filename = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
    
    np.save(npy_filename, noisy_image)
    plt.imsave(png_filename, noisy_image, cmap='gray', origin='lower', vmin=70, vmax=400)
    
    dataset_records.append({
        'image_path': npy_filename,
        'is_lens': is_lens,
        'magnitude': proxy_magnitude, 
        'arc_size': calculated_size,               
        'blended': is_blended,
        'split': split_group # Tracks which subset this image belongs to
    })

df = pd.DataFrame(dataset_records)
df.to_csv("dspl_test_metadata.csv", index=False)
print("Success! Created a balanced batch of 600 split catalog items.")

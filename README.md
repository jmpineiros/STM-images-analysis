# STM-images-analysis


Introduction: 

This is a general version of the Python script to analyze STM images of a single (stepped) crystals. The script uses a large fraction of the available data in STM images containing multiple steps and represents the data in multiple views. The script identifies terraces, steps and step edges coordinates and extracts valuable statistical information.

Some parameters are specific for every single crystal such as unit cell (unit_cell_a), interatomic distance in x (scale_atom) , interatomic distance in y (scale_atom_y), and step height (step_height), depending on the terrace and step type. An array with the atomic grid needs to be created and uploaded as well. Some parameters such as the border and gradient mark need to be chosen by the user per image, in such a way that the average step height is within a tolerance of 25% with respect to the experimental value. If that is not the case we suggest that the border parameter and/or gradient mark need to be changed and repeat the entire procedure.

The correction of grid-fitted step edges written here is specific for a (100) single crystal with A-type steps. If you want to use it for a different type of crystal you need to define an specific atomic grid and specific corrections.

If you use this script for any publication please cite the article published on Applied Surface Science, Volume 567, 30 November 2021, 150821 (https://www.sciencedirect.com/science/article/pii/S0169433221018845)

________________________________________
Usage:

Step1. Load the 'raw_data_file.txt' file. The file should have the 3D coordinates of the crystal surface and a .txt extension

Step2. Define crystals parameters in meters, valid kink angle in degrees, border and gradient mark, and temperature in Kelvin 

Step3. Use the square atomic grid available on the repository or create a dataframe with the 2D coordinates of the atomic grid specific to your crystal. Save it on the same folder with a ‘file_name’.p extension and load it.

Step4. Check step height percentage and if it is not around 25% redefine border and gradient mark
________________________________________
Important Notes:
1.	This script runs in the Python3 environment (Python 3.7).

2.	The script depends on the following python modules and packages. You need to install them and import some libraries:

       scipy - https://www.scipy.org/
       
       numpy - https://numpy.org/
       
       matplotlib - http://matplotlib.sourceforge.net/
       
       pandas - https://pandas.pydata.org/

3.	Plotting the atomic grid is a time-consuming process, which may cost minutes.




________________________________________

EXAMPLE:

- Ag (100) crystal with A steps

The raw data and the atomic grid are included in the repository. This image is not the ideal image that should be used to extract valuable statystical information such as terrace widht or kink formation energy. The image attached as raw data is just an example to show how the script works. It would be better to find an image with terraces better defined, where it is easier to separate terraces from steps on the 'z' gradient plot.

- Files

raw_data_file: Ag_20161110_m11.txt

atomic_grid_file: square_grid.py

- Parameters

border = 4.5e-14

gradient_mark= 4.8e-12

unit_cell_a= 4.0853e-10 #[m]

scale_atom = (np.sqrt(2)/2)*(4.0853e-10)  # [m]  for Ag  (100) in x

scale_atom_y=(np.sqrt(2)/2)*(4.0853e-10)  # [m]  for Ag  (100) in y

step_height= unit_cell_a/2 # [m]  for Ag (100) literature:2.4e-10 m

kink_angle= 45 # [degrees]

temp = 300  #[K] (kink formation energy section)

________________________________________

Author: Jessika M. Pineiros Bastidas, Leiden University

Email: ysikpr@gmail.com

       l.juurlink@chem.leidenuniv.nl
       
Date of last version: 2025/01/12

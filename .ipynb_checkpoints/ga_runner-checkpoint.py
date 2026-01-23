# ga_runner.py
import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd

import shutil
from joblib import Parallel, delayed 

from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import curve_fit

import torch
from DenseNet3D import DenseNet

from scipy.ndimage import binary_dilation, binary_closing
from skimage.morphology import ball

import warnings
warnings.filterwarnings('ignore')

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke_test', action='store_true', help='Run smoke test only')
    return vars(parser.parse_args())

root_dir = os.getcwd()

def create_unit_cell_copy(i, j, k, cell_type):
    if cell_type == 0:
        unit_cell = bcc
    elif cell_type == 1:
        unit_cell = fcc
    elif cell_type == 2:
        unit_cell = ot
    elif cell_type == 3:
        unit_cell = sc
    else: 
        unit_cell = dia

    unit_cell_copy = mesh.Mesh(unit_cell.data.copy())
    unit_cell_copy.x -= i * dx
    unit_cell_copy.y -= j * dy
    unit_cell_copy.z -= k * dz
    return unit_cell_copy
    
def read_matrices_from_folder(folder_path, indices_of_highest_values):
    matrices = []
    for index in indices_of_highest_values:
        file_name = f'matrix_{index}.npy'
        file_path = os.path.join(folder_path, file_name)
        matrix = np.load(file_path)
        matrices.append(matrix)
    return matrices

def cross_over(matrix1, matrix2, random_seed: random.Random = None):
    '''
    Perform a crossover between two parent lattices and produce offspring. 
    Assumes parent matrices are 3D numpy arrays (4x4x4 in thos case) with integer encoding of cell types.
    NO MUTATION at this step.
    - matrix1: 1st parental matrix
    - matrix2: 2nd parental matrix
    - random_seed: optional random state fix
    Returns the list of offspring (numpy arrays).
    '''

    if random_seed is None:
        random_seed = random

    def _copy(arr):
        return np.copy(arr)
        
    n = random_seed.randint(1, 3)
    children = []

    # Axis 0
    child1_axis0 = np.concatenate((_copy(matrix1)[:n], _copy(matrix2)[n:]), axis=0)
    child2_axis0 = np.concatenate((_copy(matrix2)[:n], _copy(matrix1)[n:]), axis=0)
    children.extend([child1_axis0, child2_axis0])

    # Axis 1 
    child1_axis1 = np.concatenate((_copy(matrix1)[:, :n], _copy(matrix2)[:, n:]), axis=1)
    child2_axis1 = np.concatenate((_copy(matrix2)[:, :n], _copy(matrix1)[:, n:]), axis=1)
    children.extend([child1_axis1, child2_axis1])

    # Axis 2 
    child1_axis2 = np.concatenate((_copy(matrix1)[:, :, :n], _copy(matrix2)[:, :, n:]), axis=2)
    child2_axis2 = np.concatenate((_copy(matrix2)[:, :, :n], _copy(matrix1)[:, :, n:]), axis=2)
    children.extend([child1_axis2, child2_axis2])
    
    return children

def read_child_matrix(file_path):
    return np.load(file_path)

def sort_by_index(filename):
    return int(filename.split('_')[1].split('.')[0])
    
def top_k_mean(modulus_list, k=1):
    flat = np.concatenate([np.asarray(g).ravel() for g in modulus_list if np.asarray(g).size>0])
    if flat.size == 0:
        return 0.0
    k = min(k, flat.size)
    topk_mean = float(np.mean(np.sort(flat)[-k:]))
    return topk_mean

def assemble_supercell(matrix,
                       unit_cells: dict,
                       permute: tuple = (0, 1, 2),
                       flip_axes: tuple = (False, False, False)):
                       
    matrix = np.transpose(matrix, permute)

    if flip_axes[0]:
        matrix = matrix[::-1, :, :]
    if flip_axes[1]:
        matrix = matrix[:, ::-1, :]
    if flip_axes[2]:
        matrix = matrix[:, :, ::-1]

    nz, nx, ny = matrix.shape
    U = unit_cell_size_expected
    super_shape = (nz * U, nx * U, ny * U)
    supercell = np.zeros(super_shape, dtype=np.uint8)

    for z in range(nz):
        for x in range(nx):
            for y in range(ny):
                cell_type = int(matrix[z, x, y])
                try:
                    unit = unit_cells[cell_type]
                except KeyError:
                    raise KeyError(f'Unknown type {cell_type} at z={z}, x={x},y={y}')
                z0, z1 = z * U, (z + 1) * U
                x0, x1 = x * U, (x + 1) * U
                y0, y1 = y * U, (y + 1) * U
                supercell[z0:z1, x0:x1, y0:y1] = unit

    return supercell
    
def seam_mask_for_supercell(S, U=41):
    mask = np.zeros((S,S,S), dtype=bool)
    step = U
    for z in range(1,4):  
        z_idx = z*step
        mask[z_idx-1:z_idx+1,:,:] = True
    for x in range(1,4):
        x_idx = x*step
        mask[:,x_idx-1:x_idx+1,:] = True
    for y in range(1,4):
        y_idx = y*step
        mask[:,:,y_idx-1:y_idx+1] = True
    return mask
    
cell_type_to_filename = {0: os.path.join(root_dir, 'unit_cells/bcc.npy'),
                         1: os.path.join(root_dir, 'unit_cells/fcc.npy'),
                         2: os.path.join(root_dir, 'unit_cells/ot.npy'),
                         3: os.path.join(root_dir, 'unit_cells/sc.npy'),
                         4: os.path.join(root_dir, 'unit_cells/dia.npy')}

unit_cell_size_expected = 41

def load_unit_cells(npy_directory: str, mapping: dict):
    unit_cells = {}
    for cell_type, fname in mapping.items():
        path = os.path.join(npy_directory, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Unit-cell file for type {cell_type} not found: {path}')
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f'Unit cell {path} must be a 3D array, got shape {arr.shape}')
        if arr.shape != (unit_cell_size_expected, unit_cell_size_expected, unit_cell_size_expected):
            raise ValueError(f'Unit cell {path} has wrong shape {arr.shape}, expected {(unit_cell_size_expected,)*3}')
        unit_cells[cell_type] = (arr != 0).astype(np.uint8)
    return unit_cells

nx, ny, nz = 4, 4, 4
dx, dy, dz = 2.5, 2.5, 2.5
array_size = 163  
n_generations = 75

model = DenseNet()                        
state_dict = torch.load(os.path.join(root_dir, 'model_state.pth'), map_location=torch.device('cpu'))
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
    
def run_short_ga(n_generations=25, mutation_rate=0.5, n_parents=100, offspring_mut_frac=0.05, iteration=0, run=0, k=1): 
    '''
    Performs a shorter version of GA optimization of lattice composition. 
    This GA does not involve any FFT simulations, the elastic modulus estimation is done with DenseNet trained on a combined dataset of random and optimized lattices.
    - n_generations: number of generations to run GA for
    - mutation_rate: hyperparameter controlling the fraction of cells inside the offspring lattice to be mutated
    - n_parents: hyperparameter controlling the number of parents selected at each iteration to produce offspring
    - offspring_mut_frac: hyperparameter controlling the fraction of offspring to be selected for mutation
    - iteration: number of BO iteration
    - run: number of GA run with the same hps
    - k: controls the function output by returning average between k best observed values
    Returns the maximum found value of specific elastic modulus (k=1) or average between best found values (k>1).
    '''
    
    print(f'Running iteration {iteration} run {run}: mutation rate = {mutation_rate}, n_parents = {n_parents}, mutation_frac = {offspring_mut_frac}')
    root_dir = os.getcwd()
    if not os.path.exists(f'ga_iteration_{iteration}_run_{run}'):
        os.makedirs(f'ga_iteration_{iteration}_run_{run}')
    os.chdir(f'ga_iteration_{iteration}_run_{run}')
    cur_dir = os.getcwd()
    
    modulus_list = []
    indices_of_highest_values_storage = []
    
    for generation in range(n_generations):
        ###                          Calculate elastic modulus and normalize by solid fraction
        if generation == 0:            
            modulus_array = np.load(os.path.join(root_dir, 'specific_E.npy'))
            modulus_list.append(modulus_array)                                                                       
        else:
            modulus_array = np.load(os.path.join(cur_dir, f'predictions/predictions_{generation}.npy'))       
        if generation != 0:
            modulus_array = np.concatenate((modulus_array, modulus_list[generation-1][indices_of_highest_values_storage[generation-1]])) # 300 + 100 = 400    
    
        ###                                     Selecting best N lattices
        n = n_parents         
        indices_of_highest_values = np.argsort(modulus_array)[-n:]
        indices_of_highest_values_storage.append(indices_of_highest_values)    
    
        if generation == 0:
            folder_path = os.path.join(root_dir, 'initial_population')
        else:
            folder_path = os.path.join(cur_dir, f'matrices_{generation}')
        matrices = read_matrices_from_folder(folder_path, indices_of_highest_values)
        random.shuffle(matrices)
        children_matrices = []

        n_cells = int(round(mutation_rate * 64))
        i = 0
        while i < len(matrices):
            if i + 1 < len(matrices):
                parent1 = matrices[i]
                parent2 = matrices[i + 1]
                i += 2
            else:
                parent1 = matrices[i]
                partner_idx = random.randint(0, len(matrices) - 2)  
                parent2 = matrices[partner_idx]
                i += 1

            children = cross_over(parent1, parent2)
            children_matrices.extend(children)   
            
        ###              Mutation specified by hyperparameters
        total_children = len(children_matrices)
        n_to_mutate = int(round(offspring_mut_frac * total_children))

        if n_to_mutate > 0 and n_cells > 0:
            idxs = random.sample(range(total_children), k=n_to_mutate)

            for idx in idxs:
                child = children_matrices[idx]
                n_mut = min(n_cells, child.size)

                inner_idxs = (range(child.size) if n_mut >= child.size else random.sample(range(child.size), k=n_mut))

                for idx in inner_idxs:
                    i, j, k = np.unravel_index(idx, child.shape)
                    child[i, j, k] = random.randint(0, 4)
      
        ###                                        Save children as matrices 
        child_directory = os.path.join(cur_dir, f'matrices_{generation+1}')
        if not os.path.exists(child_directory):
            os.makedirs(child_directory)

        for k, child_matrix in enumerate(children_matrices):
            file_path = os.path.join(child_directory, f'matrix_{k}.npy')
            np.save(file_path, child_matrix)
            
        ###                                    Generating binary representation matrices
    
        os.chdir(cur_dir)
        npy_directory = child_directory
        density_w = []
        for num, file_name in enumerate(sorted(os.listdir(npy_directory), key=lambda x: int(x.split('_')[1].split('.')[0]))):
            if file_name.endswith('.npy'):          
                file_path = os.path.join(npy_directory, file_name)
                child_matrix = read_child_matrix(file_path)  # expected shape [nz, nx, ny], ints in 0..4
                nz, nx, ny = child_matrix.shape
                expected_blocks = 4
                unit_cells_cache = load_unit_cells(npy_directory, cell_type_to_filename)
                mat = assemble_supercell(child_matrix, unit_cells_cache, permute=(1,2,0), flip_axes=(True,False,False))

                weighted_sf = float(mat.astype(bool).sum()) / float(mat.size)

                mask = seam_mask_for_supercell(164, U=41)
                selem = ball(1)
                dil_full = binary_dilation(mat, structure=selem)
                mat[mask] = dil_full[mask]

                mat = binary_closing(mat, structure=ball(1))               
            
                output_path_binary = os.path.join(cur_dir, f'binary_matrix_{generation+1}')           
                if not os.path.exists(output_path_binary):
                    os.makedirs(output_path_binary)
                np.save(os.path.join(output_path_binary, f'matrix_{num}.npy'), mat)
             
                
                density_w.append(weighted_sf)  
                if not os.path.exists(os.path.join(cur_dir, 'densities')):
                    os.makedirs(os.path.join(cur_dir, 'densities'))
                np.save(os.path.join(cur_dir, f'densities/density_w_generation_{generation+1}.npy'), np.array(density_w))
            
        ###                           Estimation of new structures with trained DenseNet       
        model.eval()  
        output_predictions = []
        folder_path = os.path.join(cur_dir, f'binary_matrix_{generation+1}')
        files = sorted(
            [f for f in os.listdir(folder_path) if f.startswith('matrix_') and f.endswith('.npy')],
            key=lambda x: int(x.split('_')[1].split('.')[0]))


        for file in files:
            file_path = os.path.join(folder_path, file)
            numpy_matrix = np.load(file_path)
            tensor_input = torch.tensor(numpy_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
            tensor_input = tensor_input.to(device)

            with torch.no_grad():
                prediction = model(tensor_input)

            prediction_numpy = prediction.cpu().numpy()
            output_predictions.append(prediction_numpy)

        output_predictions = np.array(output_predictions)            
        pred_flat = output_predictions.flatten()    
        
        if os.path.exists(os.path.join(cur_dir, f'binary_matrix_{generation+1}')):
            shutil.rmtree(os.path.join(cur_dir, f'binary_matrix_{generation+1}'))
  
        if not os.path.exists(os.path.join(cur_dir, 'predictions')):
            os.makedirs(os.path.join(cur_dir, 'predictions'))        
        np.save(os.path.join(cur_dir, f'predictions/predictions_{generation+1}.npy'), pred_flat)      
        
        ###                                 Adding parents from previous generation       
        if generation == 0:
            init_matrices_dir = os.path.join(root_dir, 'initial_population')
        else:
            init_matrices_dir = os.path.join(cur_dir, f'matrices_{generation}')
        matrices_1_dir = os.path.join(cur_dir, f'matrices_{generation+1}')
    
        init_matrices_files = sorted(os.listdir(init_matrices_dir), key=sort_by_index)
        existing_files = [int(file.split('_')[1].split('.')[0]) for file in os.listdir(matrices_1_dir) if file.startswith('matrix_')]
        next_index = max(existing_files) + 1 if existing_files else 0
        for index in indices_of_highest_values:
            if index < len(init_matrices_files):
                source_file = init_matrices_files[index]
                dest_file = f'matrix_{next_index}.npy'
                shutil.copyfile(os.path.join(init_matrices_dir, source_file), os.path.join(matrices_1_dir, dest_file))
                next_index += 1
            
                
        modulus_list.append(modulus_array)
        np.save(os.path.join(cur_dir,'modulus_list.npy'), np.array(modulus_list, dtype=object))
        np.save(os.path.join(cur_dir, 'indices_of_highest_values_storage.npy'), np.array(indices_of_highest_values_storage, dtype=object))               
            
            
    best_val = top_k_mean(modulus_list, k = 1)
    print(f'Iteration {iteration} run {run} - best value is {best_val}')
    os.chdir(root_dir)
    if os.path.exists(os.path.join(root_dir, f'ga_iteration_{iteration}_run_{run}')):
        shutil.rmtree(os.path.join(root_dir, f'ga_iteration_{iteration}_run_{run}'))
    return best_val                       


if __name__ == "__main__":
    print('Running a smoke test')
    run_short_ga(n_generations=3, mutation_rate=0.5, n_parents=10, offspring_mut_frac=0.05, iteration='test', run='test', k = 1)
    print('Completed')
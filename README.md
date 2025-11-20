# Project: Transformer Model to Predict Sequences

This project was part of the course "Introduction to Scientific Computations" at NTNU. It was developed by Aslak Vidarsson Homme and Hallstein Olesønn Hagaseth.

## Project Goal

The goal of this project was to implement a transformer model in Python, without using any AI libraries. Only NumPy and Matplotlib were used. The final results are presented in the notebook **FINAL_DRAFT_PROJECT2.ipynb**.

## Methodology

- Implemented key components of a transformer model to predict sequences of numbers.
- Compared two optimization algorithms: **Steepest Descent** and **Adam**.
- Trained the networks on 2,500 data points covering all possible sequences of length 5 with two possible values.

## Results

### Sequence Sorting
- Before training: None of the sequences were correctly sorted.  
- After training with Adam: **99.84%** of sequences were correctly sorted.  
- After training with Steepest Descent: **1.64%** of sequences were correctly sorted.

### Addition of Two Two-Digit Numbers
- Adam optimizer quickly converged below the tolerance of 0.01.  
- Steepest Descent improved gradually but did not reach high accuracy.  
- Final accuracy:  
  - Adam: **99.49%**  
  - Steepest Descent: **0.99%**

These results align with the theory that the Adam optimizer converges faster for certain problems, while Steepest Descent may be more suitable for new or different tasks.

## Technologies Used

- Python  
- NumPy  
- Matplotlib  
- Jupyter Notebook  

## How to Explore

The full implementation and results are in **FINAL_DRAFT_PROJECT2.ipynb**.  
No execution is required to understand the results; all code, plots, and analysis are included in the notebook.

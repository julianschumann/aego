# Autoencoder enabled global optimization
In this repository, the code and data used to generate the results from the paper "A machine learning approach for fighting the curse of
dimensionality in global optimization" can be found. This should allow the reconstruction of the results visible in Figures 3, 5, 6, and C6.

# Figure 3
[Fig. 3](https://github.com/julianschumann/aego/blob/main/Figure_3.pdf) shows the results of optimizing the benchmark functions, and can be produced with the code found in the subdirectories of [Benchmark_functions](https://github.com/julianschumann/aego/tree/main/Benchmark%20functions). When reproducing the results from $c_1$, $c_2$, $c_3$, and $c_4$, it should be enough to go to the respective folder and just run the repsective python files:
```
cd C_1 && python benchmark_test_c1.py
cd C_2 && python benchmark_test_c2.py 
cd C_3 && python benchmark_test_c3.py 
cd C_4 && python benchmark_test_c4.py 
```

The Figure can then be reproduced by runing [Figure_3.tex](https://github.com/julianschumann/aego/blob/main/Figure_3.tex).


# Figure 5
[Fig. 5](https://github.com/julianschumann/ae-opt/blob/main/Figure_5.pdf) is produced by running the following python files in [Results_compression](https://github.com/julianschumann/aego/tree/main/Fracture%20anisotropy%20maximization/Results_compression) in the following order. Some of the code can be run on multiple processors, by using the mpi4py package, with the number of preocessors being used being indicated:
```
python Generate_training_samples.py (150 processors)
python Generate_decoder_dimension.py 
python PCA_test.py
python Generate_decoder.py
python Run_optimization.py (150 processors)
python Comparison.py (150 processors)
python Analysis.py
```
The same then has to be done for [Results_expansion](https://github.com/julianschumann/aego/tree/main/Fracture%20anisotropy%20maximization/Results_expansion) and [Results_shear](https://github.com/julianschumann/aego/tree/main/Fracture%20anisotropy%20maximization/Results_shear)

The Figure can then be reproduced by runing [Figure_5.tex](https://github.com/julianschumann/aego/blob/main/Figure_5.tex).

It has to be noted that due to the randomness invovled in the algorithm, changing the number of processors used can change the final results achieved by running this code.

# Figure 6
[Fig. 6](https://github.com/julianschumann/aego/blob/main/Figure_6.pdf) is created by simply relying on the results produced during the creation of Fig. 5. It can then be reproduced by runing [Figure_6.tex](https://github.com/julianschumann/aego/blob/main/Figure_6.tex).

# Figure C6
[Fig. C6](https://github.com/julianschumann/aego/blob/main/Figure_12.pdf) is created by simply relying on the results produced during the creation of Fig. 3. It can then be reproduced by runing [Figure_12.tex](https://github.com/julianschumann/aego/blob/main/Figure_12.tex).

# Software used
The result show in the paper have been produced using the following software implementation:
- Python: 3.7.9
  - Numpy: 1.18.5
  - Scipy: 1.4.1
  - Tensorflow: 2.3.0
  - Keras: 2.7.0
  - Mpi4py: 3.1.3

The hybrida package used for the solving of the fracture anisotropy maximization problem can be made available on request.

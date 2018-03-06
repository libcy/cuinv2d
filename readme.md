# cuinv2d

Cuda based 2D elastic full waveform inversion program. [fd2d-adjoint](https://github.com/phlos/fd2d-adjoint) and [seisflows](https://github.com/rmodrak/seisflows) gave me many useful references in writing this. Finite difference method can be accelerated significantly by GPU, and one forward calculation is usually done within a second. The whole inversion process is also performed in GPU, which eliminates the time of reading/writing files and data transfer between host and device.

#### Compilation
```
nvcc cuinv2d.cu -arch=sm_50 -lcublas -lcusolver -lcufft
```

#### Input file
Fortran style input file which is similar to [specfem2d](https://github.com/geodynamics/specfem2d). Default input file is config in the root directory, you can specify the path through command line argument (e.g. ./a.out -c sample/config/checker_4x4). Model, source and station data of specfem2d can be used directly without any modification. Seismograms are read/written in [Seismic Unix](http://www.cwp.mines.edu/cwpcodes/) format.

command line arguments (optional, overwrites input file)
```
-c <path-to-config-file>
-o <output-directory>
-m <simulation-mode>
-p <display-mode>

-mi <initial-model-directory>
-mt <true-model-directory>
-mp <homogeneous-initial-model-vp>
-ms <homogeneous-initial-model-vs>
-mr <homogeneous-initial-model-rho>

-ns <source-number>
-nr <receiver-number>
-ni <inversion-iteration>

-as <source-alignment>
-ar <receiver-alignment>
```

#### Plotting
(Requires scipy and numpy, script modified from [seisflows](http://seisflows.readthedocs.io/en/latest/instructions_remote.html?highlight=plot))
```
./plot2d <output_directory> <parameter_name>
```
e.g.
````
./plot2d output/0005 vs
````

#### Forward modeling
The program calculates wavefield propagation with 4th order finite difference method. A unique feature of it's fdm solver is that it can be configured to run multiple forward calculations simutaneously in a single grid, so that it can make full use of the GPU regardless of the size of the model.

#### Objective function
Obective function is chosen from RMS misfit and Envelop misfit. Envelope misfit generally shows a better convergence, at a cost of ~10% more time consumption (200×200 grids, 5000 timestep).

* Comparison of envelope misfit and rms misfit (Vs_init = 2800km/s)<br>
  model_3<br>
  ![](https://raw.githubusercontent.com/libcy/cuinv2d/master/img/cm3.png) <br>
  model_7<br>
  ![](https://raw.githubusercontent.com/libcy/cuinv2d/master/img/cm7.png) <br>

#### Nonlinear optimization
Currently supports conjugate gradient method and l-bfgs method. The latter generally requires less forward calculations to complete an inversion, but there are circumstances when cg yields a better final result.

* Comparison of NLCG and L-BFGS (model_1)<br>
  Vs_init = 3500km/s<br>
  ![](https://raw.githubusercontent.com/libcy/cuinv2d/master/img/c3500.png) <br>
  Vs_init = 2800km/s<br>
  ![](https://raw.githubusercontent.com/libcy/cuinv2d/master/img/c2800.png) <br>

#### Some synthetic tests
* True model(Vs ranging from 3150km/s to 3850km/s, 25 sources and 132 stations locating randomly)<br>
![](https://raw.githubusercontent.com/libcy/cuinv2d/master/img/init.png)

* Model after 15 iterations(initial model: Vs_init = 3150km/s)<br>
![](https://raw.githubusercontent.com/libcy/cuinv2d/master/img/15.png)

#### Contact
ccy@pku.edu.cn

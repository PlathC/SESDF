<p align="center">
  <h1 align="center"><a href="https://plathc.github.io/sesdf/sesdf.html">Efficient GPU computation of large protein Solvent-Excluded Surface</a></h1>
  <a href="https://plathc.github.io/sesdf/sesdf.html">
    <img src="img/teaser.png" alt="Logo" width="100%">
  </a>
  </p>
</p>

</br>
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#requirements">Requirements</a>
    </li>
    <li>
      <a href="#use-as-a-library">Use as a library</a>
    </li>
    <li>
      <a href="#running-the-experiments">Running the experiments</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#references">References</a>
    </li>
  </ol>
</details>

This repository hosts the implementation of our paper "Efficient GPU computation of large protein Solvent-Excluded Surface".

It aims at providing a reference implementation of both the complete surface and exterior only computation. We also include the benchmarking code used for the generation of performance-related figures and tables of the paper. 

## Requirements

This project has been tested on Windows with Visual Studio 2019 and 2022 and depends on C++17, CMake, OpenGL 4.5 and CUDA (tested on CUDA 11.8). While it may work on other platform, it has not been tested for it.

Note that at least CUDA 11.8 is required for RTX40xx GPUs, and that, due to the use of deprecated functions in Contour Buildup implementation, CUDA 11 is required to run the experiments.

Finally, the memory consumption experiment relies on CUPTI which must be [installed with the CUDA Toolkit](https://developer.nvidia.com/cupti). The required library is then located under the folder: `NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64`.

## Use as a Library

If you only intend to use the project as a library to generate the molecular surface of protein, it can directly be included to another project with:

```cmake
add_subdirectory(SESDF)
target_link_libraries(<TargetName> PRIVATE sesdf)
```

## Running the experiments 

Four targets are provided and allow the generation of the data used for the following figures in the paper:

- `BCSBenchmarkCBSESDF`: Computation benchmarks in Table 2.
- `BCSCirclesAnalysis`: Circle ratios analysis in Figure 11.
- `BCSDetailedSESDF`: Per-stage detailed GPU benchmarks in Figure 14.
- `BCSMemoryConsumption`: Memory consumption queries in Table 2.

We rely on CMake for its configuration. Hence, all provided executables can be built on Windows with:

```
git clone --recursive https://github.com/PlathC/SESDF.git
cd SESDF
mkdir out
cmake -S . -B out -D SESDF_WITH_ANALYSIS=ON
cmake --build out --target "BCSBenchmarkCBSESDF" --config "Release"
cmake --build out --target "BCSCirclesAnalysis" --config "Release"
cmake --build out --target "BCSDetailedSESDF" --config "Release"
cmake --build out --target "BCSMemoryConsumption" --config "Release"
```

Experiments executables can finally be found under the folder `out/bin/`.

## Organisation of the project

The project contains a dedicated folder for each of the presented methods. These folders contain the following files:

- `NameOfTheMethod`: Main utility containing the handling of pre-allocation, building method calls and final graphics buffer access.
- `operations`: Contains every building stages which can be called by the host and delimiting their scopes as well as kernels.
- `data`: Main data required by kernels.
- `graphics`: Output data returned by the main utility class once built.

The folder `cuda` contains essential API and memory handling (in `memory` and `utils`),  some of the paper's equations (in `circles`) and general
GPU utilities (in `grid`, `setup`, `execution`, `math` and `memory`).

## Citation

If you use this code, please cite the following BibTeX:

```bibtex
@article{PlateauHolleville2024,
    author={Plateau—Holleville, Cyprien and Maria, Maxime and Mérillou, Stéphane and Montes, Matthieu},
    journal={IEEE Transactions on Visualization and Computer Graphics}, 
    title={Efficient GPU computation of large protein Solvent-Excluded Surface}, 
    year={2024},
    volume={},
    number={},
    pages={1-12},
    doi={10.1109/TVCG.2024.3380100}
}
```

## References

The code used for the reference GPU Contour-Buildup is [Gralka et al. Megamol](https://doi.org/10.1140/epjst/e2019-800167-5)'s implementation and can be found at 
[the following adress](https://github.com/UniStuttgart-VISUS/megamol/tree/master/plugins/protein_cuda/). We extracted its minimal dependencies such as part of [GLOWL](https://github.com/invor/glowl) in the same folder. 
The reference implementation of [Quan et al.'s paper](https://doi.org/10.1016/j.jcp.2016.07.007) at the basis of this work can be found [at the following address](https://github.com/quanchaoyu/MolSurfComp).

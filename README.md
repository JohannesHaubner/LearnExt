[![MIT License](https://img.shields.io/github/license/JohannesHaubner/LearnExt)](https://choosealicense.com/licenses/mit/)

# LearnExt

Code repository for the manuscript
> J. Haubner, O. Hellan, M. Zeinhofer, M. Kuchta (2024). Learning Mesh Motion Techniques with Application to Fluid-Structure Interaction, Comput. Methods Appl. Mech. Eng., 424. https://doi.org/10.1016/j.cma.2024.116890

## Usage/Examples

The Dockerfile can be used by running:
```
docker build -t learnext .
docker run -it learnext
```
or
```
docker pull ghcr.io/johanneshaubner/learnext
docker run -it ghcr.io/johanneshaubner/learnext
```

The code in learnExt/NeuralNet builds on https://github.com/sebastkm/hybrid-fem-nn.

In the Docker container, run the commands

```
cd example
python3 example_FSIbenchmarkII_learned.py
```

To redo the learning process for the hybrid PDE-NN approach run 
```
python3 example_learn.py
```
(and modify the file accordingly).

## Citation

```
@article{
title = {Learning mesh motion techniques with application to fluid–structure interaction},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {424},
pages = {116890},
year = {2024},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2024.116890},
url = {https://www.sciencedirect.com/science/article/pii/S0045782524001464},
author = {Johannes Haubner and Ottar Hellan and Marius Zeinhofer and Miroslav Kuchta},
}
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

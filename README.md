[![MIT License](https://img.shields.io/github/license/JohannesHaubner/LearnExt)](https://choosealicense.com/licenses/mit/)

# LearnExt

Code repository for the manuscript
> J. Haubner, O. Hellan, M. Zeinhofer, M. Kuchta: Learning Mesh Motion Techniques with Application to Fluid-Structure Interaction, arXiv preprint arXiv:2206.02217

## Usage/Examples

Requires a recent master version of dolfin with MeshView support. It might require the changes propsed in https://bitbucket.org/fenics-project/dolfin/issues/1123/assemble-on-mixed-meshview-forms-returns.
Moreover, it also requires a dolfin-adjoint version which supports the changes of git+https://github.com/dolfin-adjoint/pyadjoint.git@constant-adjfloat.

The Dockerfile (preliminary version) can be used by running:
```
docker build -t learnext .
docker run -it learnext
```
or
```
docker pull ghcr.io/johanneshaubner/learnext:v0.0.3
docker run -it ghcr.io/johanneshaubner/learnext:v0.0.3
```

The code in learnExt/NeuralNet builds on https://github.com/sebastkm/hybrid-fem-nn. The code in example/example_turtleFSI.py builds on the example presented in https://github.com/KVSlab/turtleFSI.

In the Docker container, run the commands

```
cd example
python3 example_FSIbenchmarkII_learned.py
```

To redo the learning process run 
```
python3 example_learn.py
```
(and modify the file accordingly).

## License

[MIT](https://choosealicense.com/licenses/mit/)

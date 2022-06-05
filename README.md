[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?style=plastic)](https://choosealicense/licenses/mit/)

# LearnExt

Code repository for the manuscript
> J. Haubner, M. Kuchta: Learning a Mesh Motion Technique with Application to Fluid-Structure Interaction and Shape Optimization

## Usage/Examples

Requires a recent master version of dolfin with MeshView support. It might require the changes propsed in https://bitbucket.org/fenics-project/dolfin/issues/1123/assemble-on-mixed-meshview-forms-returns.
Moreover, it also requires a dolfin-adjoint version which supports the changes of git+https://github.com/dolfin-adjoint/pyadjoint.git@constant-adjfloat.

The code in learnExt/NeuralNet builds on https://github.com/sebastkm/hybrid-fem-nn. The code in example/example_turtleFSI.py builds on the example presented in https://github.com/KVSlab/turtleFSI.

Install the required version of FEniCS and dolfin-adjoint and run the commands

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

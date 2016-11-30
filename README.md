# vmachine
An implementation of a [thinking machine][rf1].

<cite>Virie, P. (2015). (Yet) Another Theoretical Model of Thinking. arXiv preprint arXiv:1511.02455.<cite>
   [rf1]: <https://arxiv.org/abs/1511.02455>
   
### Behavior

* A generative model: the machine will learn to execute and produce most likely data sequences conditioned on past data.
* Make decision when to learn: the machine outputs a flag that tells external environment to feed more input data when it is not sure what to think next.
* Learning while executing: the machine accepts the very first data sequences as the inputs and learns from it.
   
License
----

[MIT](./LICENSE)


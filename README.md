# vmachine
An implementation of a [thinking machine][rf1].

<cite>Virie, P. (2015). (Yet) Another Theoretical Model of Thinking. arXiv preprint arXiv:1511.02455.<cite>
   [rf1]: <https://arxiv.org/abs/1511.02455>
   
### Behavior

* A generative model: the machine will learn to execute and produce most likely data sequences conditioned on a fixed sequence of past data.
* Learning while executing: like LSTMs, the machine can keep information up to an arbitary step into the past; while like condition models, it only needs to be trained stepwise. 
* The machine accepts the very first data sequences as the inputs and learns from them.
   
License
----

[MIT](./LICENSE)


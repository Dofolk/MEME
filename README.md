# MEME
Take some note:
* When using the encoder or decoder modules in torch, remember to set batch_first as True. The meaning is the first dimension is the batch size.
* The batch size of input and output must be the same. And the length for each batch can be not equal.

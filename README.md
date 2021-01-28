Translation Comparison
==============================

In this project I try to compare Bahdanau attention with attention mechanisms in Transformer. To do that, I trained a Transformer and a GRU RNN on the same dataset, with roughly the same network size, and same number of epochs, then compare the attention weights on the same phrase.

Needless to say the Transformer performed slighly better than the RNN model, but it could've performer even better if it wasn't due to the limitation in its hyperparameters (1 layer, 1 attention head).

The hyperparameters were limited so a proper comparision in attention weights would be possible, otherwise attention weights would be distributed on all attention heads, and RNN would only have one attention head. 

Check the notebooks to go through the training step by step, and to see the comparison. I am going to upload each model training with validation / beam search / and metrics on a different repo soon. Also, tomorrow, I will add the heatmap of the attention weights below.
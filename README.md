# Obstruction Removal

This is a simplified implementation of this [repo](https://github.com/alex04072000/ObstructionRemoval).
The main repo requires online optimiziation for the reflection removal and so the model is left
unfrozen. This makes it difficult to set up. This repo uses a frozen version of the 
obstruction removal model(which
doesn't require any optimization) and runs it.

The test.py file shows how to use the network.

The frozen model is [here](https://drive.google.com/file/d/1zB0W0aMzey2tCI8SX3jKJrFAeTWdO6ga/view?usp=sharing).
The test file expects the file to be in the top level, but this can be edited.

Using this frozen model requires A LOT of RAM (20+GB). But it can be run on the CPU in a reasonable amount of
time (~10 mins).

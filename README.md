# Supramolecular Variational Autoencoder for Reticular Frameworks

![GitHub Logo](/images/logo.png)

SmVAE is a multi-component variational autoencoder with modules that are in charge of encoding and decoding each part of the RFcode (edge, vertices, topology). Reticular frameworks are mapped with discrete RFcodes, transferred into continuous vectors, and then transferred back. To have the latent space organized around properties of interest, we add an extra component to the model that uses labeled data. Corresponding paper is published on Nature Machine Intelligence: https://doi.org/10.1038/s42256-020-00271-1.

Clone the repo, to generate a working enviroment download miniconda and use enviroment.yml to update:
```
git clone git@github.com/zhenpengyao/Supramolecular_VAE
conda env update --file enviroment.yml
```

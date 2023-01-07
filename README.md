# alphafold_MSA

This implementation of the Alphafold 2 algorithm was made for my Bachelor thesis and it is made to be able to run the Alphafold 2 monomer model direcly with MSAs as input and to get the predicted distances between pairs (disotgram) as an output.

### Installation guide

This installation guide was extracted from https://github.com/kalininalab/alphafold_non_docker, and modified to adapt it to out purpose.

First Miniconda should be installed as new en- vironment to run the program has to be created (recommended).

```{bash}
conda create −−name alphafold python==3.8
```

Then the environment has to be activated:

```{bash}
conda activate alphafold
```
Now the dependencies will be installed:


```{bash}
conda install −y −c conda−forge openmm==7.5.1 cudnn==8.2.1.32 cudatoolkit==11.0.3 pdbfixer==1.7
```
Now the following packages have to be installed:

```{bash}
pip install absl−py==0.13.0 biopython==1.79 chex==0.0.7 dm−haiku ==0.0.4 dm−tree ==0.1.6 immutabledict ==2.0.0 jax ==0.2.14 ml−collections ==0.1.0 numpy==1.19.5 scipy ==1.7.0 tensorflow ==2.5.0 pandas==1.3.4 tensorflow−cpu==2.5.0

pip install −−upgrade jax==0.2.14 jaxlib==0.1.69+cuda111 −f https://storage.googleapis.com/jax−releases/jax cuda releases.html

pip install matplotlib
```

Finally the parameters of the models have to be downloaded in the model paramet folder (have in mind that the size of the parameters is ∼ 3.5 GB):

```{bash}
wget −P /model paramet https://storage.googleapis.com / alphafold / alphafold params 2021 −07−14. tar
```
And then the .tar file has to be decompressed. The final model_paramet folder should contain the models in .npz format.


### Execution guide

In order to execute the program on the bash Terminal there have been established 2 variables:
- **aln msa path**: This variable refers to the folder where the MSAs are located. (Have in mind that the app will predict iteratively all the MSA’s given in the folder path.)
- **output dir**:This variable refers to the direction where the output of the app will be saved.
To run the app this command has to be run:

```{bash}
alphafold run .py −−aln msa path path −−output dir path
```


### Outputs

The program returns a folder in the **--output_path** with the name of the MSA that contains two files:
- protein.pdb (3d structure of the predicted protien)
- disotgram.pkl, a file that contains the distogram of distances between pairs of amino acids.



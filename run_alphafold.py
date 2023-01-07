######## LIBRARIES AND FUNCTIONS NEEDED ##########
import os
import numpy as np
import io
from Bio import SeqIO
import haiku as hk
import numpy as np
from absl import app
from absl import flags
from absl import logging
import pickle

FLAGS = flags.FLAGS


#Functions imported from the alphafold model
import component.parsers as parsers
import component.pipeline as pipeline
import component.protein as protein
import component.model.utils as utils
import component.model.model as model
import component.notebooks.notebook_utils as notebook_utils
import component.model.config as config
import component.extra_functions as extra

# functions 

def get_model_haiku_params(model_name: str, data_dir: str) -> hk.Params:
  """Get the Haiku parameters from a model name."""

  path = os.path.join(data_dir, f'{model_name}.npz')

  with open(path, 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)

  return utils.flat_params_to_haiku(params)


# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_string('aln_msa_path', None, 'Paths to folder where MSAs in Clustal format are located')

flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.') 


def predict_structure(aln_msa_path,aln_name,output_dir):

  ##################################################################################
  ############################## MSA PROCESSING ####################################
  ##################################################################################

  # get file and change multiple sequence aligmnet format from aln to stockholm 
  aln_file= aln_msa_path + "/"+aln_name
  stock_file = SeqIO.parse(aln_file, "clustal")
  stock_file_path=aln_file.replace(".aln",".stockholm")
 
  #save it in stockholm format
  count = SeqIO.write(stock_file,stock_file_path, "stockholm")

  f1 = open(stock_file_path)
  string_f1 = f1.read()
  msa_input=parsers.parse_stockholm(string_f1)


  #Change first name of Msa to query
  msa_input.descriptions[0] = 'query'

  ######################################################################################
  ##########################FEATURE CREATION AND MODEL RUNNING##########################
  ######################################################################################

  # Turn the raw data into model features.
  model_type_to_use = 0
  features_for_chain = {}
  feature_dict = {}
  feature_dict.update(pipeline.make_sequence_features(sequence=msa_input.sequences[0]
                                                      , description='query', num_res=len(msa_input.sequences[0])))

  feature_dict.update(notebook_utils.empty_placeholder_template_features(
        num_templates=0, num_res=len(msa_input.sequences[0])))

  feature_dict.update(pipeline.make_msa_features(msas=[msa_input]))
  features_for_chain[protein.PDB_CHAIN_IDS[0]] = feature_dict

  np_example = features_for_chain[protein.PDB_CHAIN_IDS[0]] #input of the model


  #############################Model import and prediction ##############################
  
    
  plddts = {}
  unrelaxed_proteins = {}
  pae_outputs={}
  ranking_confidences={}
  distograms= {}
  mean_plddt = {}

  model_names= ["model_1","model_2","model_3","model_4","model_5"]

  # try best model

  print("\n Now the models will be runned: \n")

  for model_name in model_names:

    print(f'Running {model_name}')
    model_name1= model_name

    cfg = config.model_config(model_name1)
    cfg.data.eval.num_ensemble = 1
    model_name='params_'+model_name1
    params = get_model_haiku_params(model_name,'./model_paramet')
    model_runner = model.RunModel(cfg, params)

    #prediction

    processed_feature_dict = model_runner.process_features(np_example, random_seed=0)
    prediction = model_runner.predict(processed_feature_dict, random_seed=0)
    print("Prediction {} successful!".format(model_name1))

    mean_plddt[model_name1] = prediction['plddt'].mean()

    ############ saving prediction output parameters #############

    distograms[model_name1] = extra.distogram_distances(prediction["distogram"])

    if 'predicted_aligned_error' in prediction:
        pae_outputs[model_name1] = (prediction['predicted_aligned_error'],
                                    prediction['max_predicted_aligned_error'])
    else:
      # Monomer models are sorted by mean pLDDT. Do not put monomer pTM models here as they
      # should never get selected.

      ranking_confidences[model_name1] = prediction['ranking_confidence']
      plddts[model_name1] = prediction['plddt']
    

    # Set the b-factors to the per-residue plddt
    final_atom_mask = prediction['structure_module']['final_atom_mask']
    b_factors = prediction['plddt'][:, None] * final_atom_mask
    unrelaxed_protein = protein.from_prediction(
        processed_feature_dict,
        prediction,
        b_factors=b_factors,
        remove_leading_feature_dimension=( model_type_to_use == 0 )) #Monomer=0
    unrelaxed_proteins[model_name1] = unrelaxed_protein

    # Delete unused outputs to save memory.
    del model_runner
    del params
    del prediction
      



  # Find the best model according to the mean pLDDT.

  best_model_name = max(mean_plddt.keys(), key=lambda x: mean_plddt[x])
  print("The best model by mean pLLDT is: " + best_model_name)

  print("Running: "+ best_model_name)

  print('\nWarning: Running without the relaxation stage.') #### without AMBER relaxation

  relaxed_pdb = protein.to_pdb(unrelaxed_proteins[best_model_name])

  print("Saving prediction...")

  #################################Save model outputs##################################
  # Create folder for prediction
  try:
    os.mkdir(output_dir+f'/{aln_name.replace(".aln","")}')
  except:
    pass


  # Write out the pdb prediction

  with open(output_dir+f'/{aln_name.replace(".aln","")}/protein.pdb', 'w') as f:
    f.write(relaxed_pdb)
    f.close()

  del relaxed_pdb

  # Save dictionary of disotgram and plddt of best model

  #create dict for best model output
  distogram = {}
  distogram["distogram"] = distograms[best_model_name]
  distogram["plddt"]= plddts[best_model_name]
  distogram["aminoacids"]=  msa_input.sequences[0]


  with open(output_dir+f'/{aln_name.replace(".aln","")}/distograms.pkl', 'wb') as f:
    pickle.dump(distogram,f)
    f.close()

  del distogram
  print("Prediction saved!")


def main(argv):

  fun = lambda x : os.path.isfile(os.path.join(FLAGS.aln_msa_path,x))
  msas = filter(fun, os.listdir(FLAGS.aln_msa_path))
  msas= [x for x in msas if x.endswith(".aln")]
  # Predict structure for each of the sequences.
  for msa  in msas:
    print("Predicting structure of: "+msa)
    predict_structure(
        aln_msa_path=FLAGS.aln_msa_path,
        aln_name= msa ,
        output_dir= FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'aln_msa_path',
      'output_dir',
  ])

  app.run(main)
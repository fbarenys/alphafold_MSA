# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the features for the AlphaFold multimer model."""

import collections
import contextlib
import copy
import dataclasses
import json
import os
import tempfile
from typing import Mapping, MutableMapping, Sequence

from absl import logging

import component.residue_constants as residue_constants
import component.feature_processing as feature_processing
import component.parsers as parsers
import component.pipeline as pipeline
import numpy as np




def convert_monomer_features(
    monomer_features: pipeline.FeatureDict,
    chain_id: str) -> pipeline.FeatureDict:
  """Reshapes and modifies monomer features for multimer models."""
  converted = {}
  converted['auth_chain_id'] = np.asarray(chain_id, dtype=np.object_)
  unnecessary_leading_dim_feats = {
      'sequence', 'domain_name', 'num_alignments', 'seq_length'}
  for feature_name, feature in monomer_features.items():
    if feature_name in unnecessary_leading_dim_feats:
      # asarray ensures it's a np.ndarray.
      feature = np.asarray(feature[0], dtype=feature.dtype)
    elif feature_name == 'aatype':
      # The multimer model performs the one-hot operation itself.
      feature = np.argmax(feature, axis=-1).astype(np.int32)
    elif feature_name == 'template_aatype':
      feature = np.argmax(feature, axis=-1).astype(np.int32)
      new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
      feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
    elif feature_name == 'template_all_atom_masks':
      feature_name = 'template_all_atom_mask'
    converted[feature_name] = feature
  return converted


def int_id_to_str_id(num: int) -> str:
  """Encodes a number as a string, using reverse spreadsheet style naming.

  Args:
    num: A positive integer.

  Returns:
    A string that encodes the positive integer using reverse spreadsheet style,
    naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
    usual way to encode chain IDs in mmCIF files.
  """
  if num <= 0:
    raise ValueError(f'Only positive integers allowed, got {num}.')

  num = num - 1  # 1-based indexing.
  output = []
  while num >= 0:
    output.append(chr(num % 26 + ord('A')))
    num = num // 26 - 1
  return ''.join(output)


def add_assembly_features(
    all_chain_features: MutableMapping[str, pipeline.FeatureDict],
    ) -> MutableMapping[str, pipeline.FeatureDict]:
  """Add features to distinguish between chains.

  Args:
    all_chain_features: A dictionary which maps chain_id to a dictionary of
      features for each chain.

  Returns:
    all_chain_features: A dictionary which maps strings of the form
      `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
      chains from a homodimer would have keys A_1 and A_2. Two chains from a
      heterodimer would have keys A_1 and B_1.
  """
  # Group the chains by sequence
  seq_to_entity_id = {}
  grouped_chains = collections.defaultdict(list)
  for chain_id, chain_features in all_chain_features.items():
    seq = str(chain_features['sequence'])
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped_chains[seq_to_entity_id[seq]].append(chain_features)

  new_all_chain_features = {}
  chain_id = 1
  for entity_id, group_chain_features in grouped_chains.items():
    for sym_id, chain_features in enumerate(group_chain_features, start=1):
      new_all_chain_features[
          f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
      seq_length = chain_features['seq_length']
      chain_features['asym_id'] = chain_id * np.ones(seq_length)
      chain_features['sym_id'] = sym_id * np.ones(seq_length)
      chain_features['entity_id'] = entity_id * np.ones(seq_length)
      chain_id += 1

  return new_all_chain_features


def pad_msa(np_example, min_num_seq):
  np_example = dict(np_example)
  num_seq = np_example['msa'].shape[0]
  if num_seq < min_num_seq:
    for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask'):
      np_example[feat] = np.pad(
          np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
    np_example['cluster_bias_mask'] = np.pad(
        np_example['cluster_bias_mask'], ((0, min_num_seq - num_seq),))
  return np_example


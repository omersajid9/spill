"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import os
import time
import torch
import argparse
import shutil
import yaml
import copy

from lighthouse.common.utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
from easydict import EasyDict

class BaseOptions(object):
    def __init__(self, model, dataset, feature, resume, domain):
        self.model = model
        self.dataset = dataset
        self.feature = feature
        self.resume = resume
        self.domain = domain
        self.opt = {}

    @property
    def option(self):
        if len(self.opt) == 0:
            raise RuntimeError('option is empty. Did you run parse()?')
        return self.opt

    def update(self, yaml_file):
        with open(yaml_file, 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            self.opt.update(yml)

    def parse(self):
        base_cfg = 'configs/base.yml'
        feature_cfg = f'configs/feature/{self.feature}.yml'
        model_cfg = f'configs/model/{self.model}.yml'
        dataset_cfg = f'configs/dataset/{self.dataset}.yml'
        cfgs = [base_cfg, feature_cfg, model_cfg, dataset_cfg]
        for cfg in cfgs:
            self.update(cfg)

        self.opt = EasyDict(self.opt)

        if self.resume:
            self.opt.results_dir = os.path.join(self.opt.results_dir, self.model, f"{self.dataset}_finetune", self.feature)
        else:
            self.opt.results_dir = os.path.join(self.opt.results_dir, self.model, self.dataset, self.feature)
            if self.domain:
                self.opt.results_dir = os.path.join(self.opt.results_dir, self.domain)

        self.opt.ckpt_filepath = os.path.join(self.opt.results_dir, self.opt.ckpt_filename)
        self.opt.train_log_filepath = os.path.join(self.opt.results_dir, self.opt.train_log_filename)
        self.opt.eval_log_filepath = os.path.join(self.opt.results_dir, self.opt.eval_log_filename)

        # feature directory
        v_feat_dirs = None
        t_feat_dir = None
        a_feat_dirs = None
        a_feat_types = None
        t_feat_dir_pretrain_eval = None

        if self.dataset == 'qvhighlight_pretrain':
            
            dataset = self.dataset.replace('_pretrain', '')

            if self.feature == 'clip_slowfast_pann':
                v_feat_dirs = [f'features/{dataset}/clip', f'features/{dataset}/slowfast']
                t_feat_dir = f'features/{dataset}/clip_text_subs_train'
                t_feat_dir_pretrain_eval = f'features/{dataset}/clip_text'
                a_feat_dirs = [f'features/{dataset}/pann']
                a_feat_types = self.opt.a_feat_types
                
            elif self.feature == 'clip_slowfast':
                v_feat_dirs = [f'features/{dataset}/clip', f'features/{dataset}/slowfast']
                t_feat_dir = f'features/{dataset}/clip_text_subs_train'
                t_feat_dir_pretrain_eval = f'features/{dataset}/clip_text'

            elif self.feature == 'clip':
                v_feat_dirs = [f'features/{dataset}/clip']
                t_feat_dir = f'features/{dataset}/clip_text_subs_train'
                t_feat_dir_pretrain_eval = f'features/{dataset}/clip_text'

            else:
                raise ValueError(f'For pre-train, features should include CLIP, but {self.feature} is used.')
        
        else:
            if self.feature == 'clip_slowfast_pann':
                v_feat_dirs = [f'features/{self.dataset}/clip', f'features/{self.dataset}/slowfast']
                t_feat_dir = f'features/{self.dataset}/clip_text'
                a_feat_dirs = [f'features/{self.dataset}/pann']
                a_feat_types = self.opt.a_feat_types
                
            elif self.feature == 'clip_slowfast':
                v_feat_dirs = [f'features/{self.dataset}/clip', f'features/{self.dataset}/slowfast']
                t_feat_dir = f'features/{self.dataset}/clip_text'

            elif self.feature == 'clip':
                v_feat_dirs = [f'features/{self.dataset}/clip']
                t_feat_dir = f'features/{self.dataset}/clip_text'

            elif self.feature == 'resnet_glove':
                v_feat_dirs = [f'features/{self.dataset}/resnet']
                t_feat_dir = f'features/{self.dataset}/glove'

            elif self.feature == 'i3d_clip':
                v_feat_dirs = [f'features/{self.dataset}/i3d']
                t_feat_dir = f'features/{self.dataset}/clip_text'

            elif self.feature == 'clap':
                a_feat_dirs = [f'features/{self.dataset}/clap']
                a_feat_types = self.opt.a_feat_types
                t_feat_dir = f'features/{self.dataset}/clap_text'

        self.opt.v_feat_dirs = v_feat_dirs
        self.opt.t_feat_dir = t_feat_dir
        self.opt.a_feat_dirs = a_feat_dirs
        self.opt.a_feat_types = a_feat_types
        self.opt.t_feat_dir_pretrain_eval = t_feat_dir_pretrain_eval

    def clean_and_makedirs(self):
        if 'results_dir' not in self.opt:
            raise RuntimeError('results_dir is not set in self.opt. Did you run parse()?')
        
        if os.path.exists(self.opt.results_dir):
            shutil.rmtree(self.opt.results_dir)

        os.makedirs(self.opt.results_dir, exist_ok=True)
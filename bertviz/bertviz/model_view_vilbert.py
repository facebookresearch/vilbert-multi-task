# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Change log
# 12/12/18  Jesse Vig   Adapted to BERT model
# 12/19/18  Jesse Vig   Assorted cleanup. Changed orientation of attention matrices. Updated comments.


"""Module for postprocessing and displaying transformer attentions.

This module is designed to be called from an ipython notebook.
"""

import json
from bertviz.attention import get_attention_vilbert, get_attention_vilbert_tasks
from IPython.core.display import display, HTML, Javascript
import os

def show(model, tokenizer, batch):
    vis_html = """
      <span style="user-select:none">
        Attention: <select id="filter">
        <option value="all">All</option>
          <option value="aa">Sentence -> Sentence </option>
          <option value="ab">Sentence -> Image </option>
          <option value="ba">Image -> Sentence </option>
          <option value="bb">Image -> Image </option>
        </select>
      </span>
      <div id='vis'></div>
    """
    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'model_view.js')).read()
    attn_data = get_attention_vilbert(model, tokenizer, batch)
    params = {
        'attention': attn_data,
        'default_filter': "bb"
    }
    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))

def show_task(model, tokenizer, batch, sents):
    vis_html = """
      <span style="user-select:none">
        Attention: <select id="filter">
        <option value="all">All</option>
          <option value="aa">Sentence -> Sentence </option>
          <option value="ab">Sentence -> Image </option>
          <option value="ba">Image -> Sentence </option>
          <option value="bb">Image -> Image </option>
        </select>
      </span>
      <div id='vis'></div>
    """
    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'model_view.js')).read()
    attn_data = get_attention_vilbert_tasks(model, tokenizer, batch, sents)
    params = {
        'attention': attn_data,
        'default_filter': "bb"
    }
    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))


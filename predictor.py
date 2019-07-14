#!/usr/bin/env python
"""
    Written by Pengfei Sun. The model is borrowed from openai/gpt2
"""
import os
import json
import fire
import regex as re
from functools import lru_cache

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

class GPTTO(object):
      def __init__(self):
          # input pre_defined param:
          #self.x = input_context
          self.init=0          

      def default_hparams(self):
          return HParams(
	        n_vocab=0,
       		n_ctx=1024,
        	n_embd=768,
        	n_head=12,
        	n_layer=12,
    		)

      def shape_list(self, x):
          static = x.shape.as_list()
          dynamic = tf.shape(x)
          
          return [dynamic[i] if s is None else s for i, s in enumerate(static)] 
      
      def softmax(self, x, axis=-1):
          x = x - tf.reduce_max(x, axis=axis, keepdims=True)
          ex= tf.exp(x)
          return ex/tf.reduce_sum(ex, axis=axis, keepdims=True)

      def gelu(self, x):
          return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

      def norm(self, x, scope, *, axis=-1, epsilon=1e-5):
          with tf.variable_scope(scope):
               n_state = x.shape[-1].value
               g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
               b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
               u = tf.reduce_mean(x, axis=axis, keepdims=True)
               s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
               x = (x - u) * tf.rsqrt(s + epsilon)
               x = x*g + b

          return x
       
      def split_states(self, x, n):
           *start, m = self.shape_list(x)
           return tf.reshape(x, start + [n, m//n])

      def merge_states(self, x):
           *start, a, b = self.shape_list(x)
           return tf.reshape(x, start + [a*b])
                
      def conv1d(self, x, scope, nf, *, w_init_stdev=0.02):
           with tf.variable_scope(scope):
                *start, nx = self.shape_list(x)
                w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
                b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
                c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
                return c
  
      def attention_mask(self, nd, ns, *, dtype):
           i = tf.range(nd)[:,None]
           j = tf.range(ns)
           m = i >= j - ns + nd
           return tf.cast(m, dtype) 

      def attn(self, x, scope, n_state, *, past, hparams):
           assert x.shape.ndims == 3  # Should be [batch, sequence, features]
           assert n_state % hparams.n_head == 0
           if past is not None:
              assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

           def split_heads(x):
        	# From [batch, sequence, features] to [batch, heads, sequence, features]
               return tf.transpose(self.split_states(x, hparams.n_head), [0, 2, 1, 3])

           def merge_heads(x):
               # Reverse of split_heads
               return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

           def mask_attn_weights(w):
        	# w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
               _, _, nd, ns = self.shape_list(w)
               b = self.attention_mask(nd, ns, dtype=w.dtype)
               b = tf.reshape(b, [1, 1, nd, ns])
               w = w*b - tf.cast(1e10, w.dtype)*(1-b)
               return w

           def multihead_attn(q, k, v):
        	# q, k, v have shape [batch, heads, sequence, features]
               w = tf.matmul(q, k, transpose_b=True)
               w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

               w = mask_attn_weights(w)
               w = self.softmax(w)
               a = tf.matmul(w, v)
               return a

           with tf.variable_scope(scope):
                c = self.conv1d(x, 'c_attn', n_state*3)
                q, k, v = map(split_heads, tf.split(c, 3, axis=2))
                present = tf.stack([k, v], axis=1)
                if past is not None:
                        pk, pv = tf.unstack(past, axis=1)
                        k = tf.concat([pk, k], axis=-2)
                        v = tf.concat([pv, v], axis=-2)
                a = multihead_attn(q, k, v)
                a = merge_heads(a)
                a = self.conv1d(a, 'c_proj', n_state)
                return a, present


      def mlp(self, x, scope, n_state, *, hparams):
           with tf.variable_scope(scope):
                nx = x.shape[-1].value
                h  = self.gelu(self.conv1d(x, 'c_fc', n_state))
                h2 = self.conv1d(h, 'c_proj', nx)
                return h2

      def block(self, x, scope, *, past, hparams):
           with tf.variable_scope(scope):
               nx = x.shape[-1].value
               a, present = self.attn(self.norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
               x = x + a
               m = self.mlp(self.norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
               x = x + m
               return x, present

      def past_shape(self, *, hparams, batch_size=None, sequence=None):
           return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

      def expand_tile(self, value, size):
           value = tf.convert_to_tensor(value, name='value')
           ndims = value.shape.ndims
           return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

      def positions_for(self, tokens, past_length):
           batch_size = tf.shape(tokens)[0]
           nsteps = tf.shape(tokens)[1]
           return self.expand_tile(past_length + tf.range(nsteps), batch_size)     
       
      def model(self, hparams, X, past=None, scope='model', reuse=False):
           with tf.variable_scope(scope, reuse=reuse):
                results = {}
                batch, sequence = self.shape_list(X)

                wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
                wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
                past_length = 0 if past is None else tf.shape(past)[-2]
                h = tf.gather(wte, X) + tf.gather(wpe, self.positions_for(X, past_length))

        	# Transformer
                presents = []
                pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
                assert len(pasts) == hparams.n_layer
                for layer, past in enumerate(pasts):
            	    h, present = self.block(h, 'h%d' % layer, past=past, hparams=hparams)
            	    presents.append(present)
                results['present'] = tf.stack(presents, axis=1)
                h = self.norm(h, 'ln_f')

     	        # Language model loss.  Do tokens <n predict token n?
                h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
                logits = tf.matmul(h_flat, wte, transpose_b=True)
                logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
                results['logits'] = logits
                return results
@lru_cache()
def bytes_to_unicode():
          bs = list(range(ord("!"), ord("~")+1))+ \
               list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
          cs = bs[:]
          n  = 0
          for b in range(2**8):
             if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
          cs = [chr(n) for n in cs]
          return dict(zip(bs, cs)) 

def get_pairs(word):
          pairs = set()
          prev_char = word[0]
          for char in word[1:]:
              pairs.add((prev_char, char))
              prev_char = char
          return pairs

class Encoder(object):
      def __init__(self, reduced_encoder, bpe_=None, 
                   byte_encoder=None, byte_decoder=None):
          self.reduce_encoder = reduced_encoder
          self.reduce_decoder = {v:k for k,v in self.reduce_encoder.items()} 
          self.byte_encoder= byte_encoder
          self.byte_decoder= byte_decoder
          if bpe_ is not None:
             self.bpe_dict = dict(zip(bpe_, range(len(bpe_))))
          self.cache = {}
          self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  
   
      def bpe(self, token):
          if token in self.cache:
            return self.cache[token]
          word = tuple(token)
          pairs = get_pairs(word)

          if not pairs:
             return token

          while True:
              bigram = min(pairs, key = lambda pair: self.bpe_dict.get(pair, float('inf')))
              if bigram not in self.bpe_dict:
                 break
              first, second = bigram
              new_word = []
              i = 0
              while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
              new_word = tuple(new_word)
              word = new_word
              if len(word) == 1:
                 break
              else:
                 pairs = get_pairs(word)
          word = ' '.join(word)
          self.cache[token] = word
          return word

      def encode(self, text):
          bpe_tokens = []
          for token in re.findall(self.pat, text):
              token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
              bpe_tokens.extend(self.reduce_encoder[token_] for token_ in self.bpe(token).split(' '))
          return bpe_tokens

      def decode(self, tokens):
          text = ''.join([self.reduce_decoder[token_] for token_ in tokens])
          text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
          return text

def get_encoder(model_path, reduced_voc):
      with open(model_path + 'encoder.json', 'r') as f:
           encoder_l = json.load(f)
      with open(model_path + 'vocab.bpe', 'r', encoding='utf-8') as f:
           bpe_data = f.read()
      bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

      reduce_dict = {}
      byte_encoder= bytes_to_unicode()
      byte_decoder= {v:k for k,v in byte_encoder.items()}
        
      index_set = []
      for index, sub_word in enumerate(reduced_voc):
          subset_ = []
          for keys, values in encoder_l.items():
              keys = bytearray([byte_decoder[c] for c in keys]).decode('utf-8', errors='replace')
              if sub_word == keys or (' '+sub_word.lower()) == keys or \
                 sub_word.lower() == keys or (' '+sub_word.lower())==keys:
                 reduce_dict[values] = keys
                 subset_.append(values)
          index_set.append(subset_)
      
      dict_index = [k for k, v in reduce_dict.items()]
      dict_index.sort()

      encoder = {k:v for k, v in encoder_l.items() if v in dict_index}
      return Encoder(
                 reduced_encoder=encoder,
                 bpe_ = bpe_merges,# could be reduced bpe
                 byte_encoder=byte_encoder,
                 byte_decoder=byte_decoder  
                 ), reduce_dict, dict_index, index_set # index_set is used for probability calculation

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def sample_sequence(*, hparams, length, reduced_dict_index,index_set,
                    start_token=None, batch_size=None,
                    context=None, temperature=1, top_k=0, GPT=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):

        lm_output = GPT.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        for i, sub_index in enumerate(index_set):
            if i ==0:
               out_logits =  tf.reduce_sum(tf.gather(logits, sub_index, axis=-1),
                                            axis=-1, keepdims=True)
            else:
               out_logits = tf.concat([out_logits, tf.reduce_sum(tf.gather(logits, sub_index, axis=-1), 
                                       axis=-1, keepdims=True)], axis=-1)
   
        presents = lm_output['present']
        presents.set_shape(GPT.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': out_logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)

    #        logits = top_k_logits(logits, k=top_k)
    #        samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
    #        samples = tf.gather(reduced_dict_index, samples)

            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                logits, 
            #    samples,
            #    tf.concat([output, samples], axis=1)
            ]

        past, logits = body(None, context, context)

        return tf.math.l2_normalize(logits)

class RTRNNLM(object):
      def __init__(self, batch_size=1, seed=1234, reduced_voc=None,
                  model_path=None):
          self.reduced_voc =  reduced_voc
          self.model_path  = model_path
          self.enc, reduced_dict, dict_index, index_set = \
                                       get_encoder(self.model_path, self.reduced_voc) # model_path
          
          model = GPTTO()
          hparams = model.default_hparams()
          with open(model_path + 'hparams.json') as f:
              hparams.override_from_dict(json.load(f))

          self.context = tf.placeholder(tf.int32, [batch_size, None])
          np.random.seed(seed)
          tf.set_random_seed(seed)
          self.output = sample_sequence(hparams=hparams,
                        length=1, reduced_dict_index=dict_index,index_set=index_set,
                        context=self.context, batch_size=1, 
                        temperature=2, top_k=20, GPT=model)
           
          saver = tf.train.Saver()
          ckpt = tf.train.latest_checkpoint(model_path)
          self.sess = tf.Session()
          saver.restore(self.sess, ckpt)

      def prob(self, prior_txt):
          context_tokens = self.enc.encode(prior_txt)
          out = self.sess.run([self.output],
                     feed_dict={self.context: [context_tokens]
                     })
          #out = out[:, len(context_tokens):]
          #text = self.enc.decode(out[0])
          return out

def predict(context):
    model_dir = "/userdata/psun/ANSR/src/language_model/src/models/117M/"
    voc = ["I", "You",  "My",  "They",  "It",  "Am", "Are", "Need", "Feel",
           "Is",  "Hungry", "Help", "Tired", "Not", "How", "Okay", "Very",
           "Thirsty", "Comfortable", "Right","Please", "Hope", "Clean",
           "Glasses", "Nurse", "Closer", "Bring", "What", "Where", "Tell",
           "That", "Going", "Music", "Like", "Outside", "Do", "Have", "Faith",
           "Success", "Coming", "Good", "Bad", "Here", "Family", "Hello", "Goodbye",
           "Computer", "Yes", "Up", "No"]    
    RNNLM = RTRNNLM(reduced_voc=voc, model_path=model_dir) 
    prob  = RNNLM.prob(context)
    return prob

if __name__ == '__main__':
   inputs = 'I need'
   print(predict(inputs))

#vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

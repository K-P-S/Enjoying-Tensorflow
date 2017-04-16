#supervised encoder/decoder model for generating music
#piano/ragtime
#credit: youtube, Sirajology
#his credit: Github, 


#dependencies
import numpy as np
import tensorflow
from deepmusic.moduleloader import ModuleLoader
#predicts next key
from deepmusic.keyboardcell import KeyboardCell
#encapsulate song data so to run get_scale, get_relative_methods
import deepmusic.songstruct as music

def build_network(self):
  #create computation graph, encapsulate session and graph initializion
  input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()
  
  #note data, midi format
  with tf.name_scope('placeholder_inputs'):
    self.inputs = [
      tf.placeholder(
      tf.float32, #numerical data
      [self.batch_size, input_dim], 
      name = 'input' #how much data
      )
    ]
    
  #define targets
  #targets 88 key, binary classification problem
  with tf.name_scope('placeholder_targets'):
    self.targets = [
      tf.placeholder(
      tf.int32, #0 or 1
      [self.batch_size],
      name = 'target')
    ]
    
  #feed hidden state (recurrent net)
  with tf.name_scope ('placeholder_use_prev'):
    self.use_prev = [
      tf.placeholder(
      tf.bool,
      [],
      name = 'use_prev')
    ]
    
  #define network
  self.loop_processing = ModuleLoader.loop_processings.build_module(self.args)
  #take previous value for next input
  def loop_rnn(prev, i):
    next_input = self.loop_processing(prev)
    return tf.cond(self.prev[i], lambda: next_input, lambda:self.inputs[i]) #returns one or other param
  #sequence to sequence model
  self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(
    decoder_inputs = self.inputs, #defined in keyboard cell
    initial_state = None,
    cell = KeyboardCell,
    loop_function = loop_rnn
  ) 
  
  #training step
  #define loss function: cross entropy
  loss_func = tf.nn.seq2seq.sequence_loss(
  self.outputs,
  self.targets,
  softmax_loss_function = tf.nn.softmax.cross_entropy_with_logits,
  average_across_timesteps = True,
  average_across_batch = True
  )
  
  #initialize adam optimizer, minimize loss
  opt = tf.train.AdamOptimizer(
    learning_rate = self.current_learning_rate,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-08
  )
  
  self.opt_op = opt.minimize(loss_func)





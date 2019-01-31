import tensorflow as tf
import numpy as np
import os

from ddflow_model import DDFlowModel
from config.extract_config import config_dict

# manually select one or several free gpu 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# autonatically select one free gpu
#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
#os.system('rm tmp')


def main(_):
    config = config_dict('./config/config.ini')
    run_config = config['run']
    dataset_config = config['dataset']    
    distillation_config = config['distillation']
    model = DDFlowModel(batch_size=run_config['batch_size'],
                      iter_steps=run_config['iter_steps'], 
                      initial_learning_rate=run_config['initial_learning_rate'],
                      decay_steps=run_config['decay_steps'],
                      decay_rate=run_config['decay_rate'],
                      is_scale=run_config['is_scale'],
                      num_input_threads=run_config['num_input_threads'],
                      buffer_size=run_config['buffer_size'],
                      beta1=run_config['beta1'],
                      num_gpus=run_config['num_gpus'],
                      save_checkpoint_interval=run_config['save_checkpoint_interval'],
                      write_summary_interval=run_config['write_summary_interval'],
                      display_log_interval=run_config['display_log_interval'],
                      allow_soft_placement=run_config['allow_soft_placement'],
                      log_device_placement=run_config['log_device_placement'],
                      regularizer_scale=run_config['regularizer_scale'],
                      cpu_device=run_config['cpu_device'],
                      save_dir=run_config['save_dir'],
                      checkpoint_dir=run_config['checkpoint_dir'],
                      model_name=run_config['model_name'],
                      sample_dir=run_config['sample_dir'],
                      summary_dir=run_config['summary_dir'],
                      training_mode=run_config['training_mode'],
                      is_restore_model=run_config['is_restore_model'],
                      restore_model=run_config['restore_model'],
                      dataset_config=dataset_config,
                      distillation_config= distillation_config
                      )
    
    if run_config['mode'] == "train":
        model.train()
    elif run_config['mode'] == 'test':
        model.test(restore_model=config['test']['restore_model'],
                   save_dir=config['test']['save_dir'])
    elif run_config['mode'] == 'generate_fake_flow_occlusion':
        model.generate_fake_flow_occlusion(restore_model=config['generate_fake_flow_occlusion']['restore_model'],
                                           save_dir=config['generate_fake_flow_occlusion']['save_dir'])
    else:
        raise ValueError('Invalid mode. Mode should be one of {train, test, generate_fake_flow_occlusion}')

if __name__ == '__main__':
    tf.app.run() 

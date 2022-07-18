from clearml.automation import PipelineController

PROJECT_NAME = 'sentiment-classification-rtmovies5c'

parameters = {
    'validation_split': 0.1,
    'seed': 42,
    'pre_trained_model': 'bert-base-uncased',
    'batch_size': 16,
    'max_length': 512,
    'lr': 2e-5,
    'freeze_backbone': True,
    'num_epochs': 3,
    'accelerator': 'auto',
    'devices': 'auto',
}

pipe = PipelineController(
    name = 'LitTransformers_pipeline',
    project = PROJECT_NAME,
    version = '0.1'
)

pipe.add_parameter('validation_split', parameters['validation_split'])
pipe.add_parameter('seed', parameters['seed'])
pipe.add_parameter('pre_trained_model', parameters['pre_trained_model'])
pipe.add_parameter('batch_size', parameters['batch_size'])
pipe.add_parameter('max_length', parameters['max_length'])
pipe.add_parameter('lr', parameters['lr'])
pipe.add_parameter('freeze_backbone', parameters['freeze_backbone'])
pipe.add_parameter('num_epochs', parameters['num_epochs'])
pipe.add_parameter('accelerator', parameters['accelerator'])
pipe.add_parameter('devices', parameters['devices'])

pipe.add_step(
    name = 'data_split',
    base_task_name='LitTransformers_pipe_1 - train/val split',
    base_task_project=PROJECT_NAME,
    parameter_override={'General/seed', '${pipeline.seed}',
                        'General/validation_split', '${pipeline.validation_split}'} # type: ignore
)

pipe.add_step(
    name = 'train_model',
    base_task_name='LitTransformers_pipe_2 - train_model',
    base_task_project=PROJECT_NAME,
    parents=['data_split'],
    parameter_override={'General/seed', '${pipeline.seed}',
                        'General/pre_trained_model', '${pipeline.pre_trained_model}',
                        'General/batch_size', '${pipeline.batch_size}',
                        'General/max_length', '${pipeline.max_length}',
                        'General/lr', '${pipeline.lr}',
                        'General/freeze_backbone', '${pipeline.freeze_backbone}',
                        'General/num_epochs', '${pipeline.num_epochs}',
                        'General/accelerator', '${pipeline.accelerator}',
                        'General/devices', '${pipeline.devices}'} # type: ignore
)

if __name__ == '__main__':
    pipe.set_default_execution_queue("default")
    pipe.start_locally()

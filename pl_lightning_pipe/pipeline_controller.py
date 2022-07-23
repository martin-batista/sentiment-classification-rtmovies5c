from clearml import Task
from clearml.automation import PipelineController
from pipe_conf import PROJECT_NAME

parameters = {
    'validation_split': 0.2,
    'seed': 42,
    'pre_trained_model': 'bert-base-uncased',
    'batch_size': 16,
    'max_length': 64,
    'lr': 2e-5,
    'num_epochs': 10,
    'stratified_sampling': True,
    'stratified_epochs': 7,
    'accelerator': 'auto',
    'devices': 'auto',
}

Task.add_requirements('requirements.txt')
pipe = PipelineController(
    name = 'pl_train_models_pipeline',
    project = PROJECT_NAME,
    version = '0.1'
)

pipe.set_default_execution_queue('GPU')

pipe.add_parameter('validation_split', parameters['validation_split'])
pipe.add_parameter('seed', parameters['seed'])
pipe.add_parameter('batch_size', parameters['batch_size'])
pipe.add_parameter('max_length', parameters['max_length'])
pipe.add_parameter('lr', parameters['lr'])
pipe.add_parameter('num_epochs', parameters['num_epochs'])
pipe.add_parameter('stratified_sampling', parameters['stratified_sampling'])
pipe.add_parameter('stratified_epochs', parameters['stratified_epochs'])
pipe.add_parameter('accelerator', parameters['accelerator'])
pipe.add_parameter('devices', parameters['devices'])

pipe.add_step(
    name = 'data_split',
    base_task_name='data_split',
    base_task_project=PROJECT_NAME,
    parameter_override={'General/seed': '${pipeline.seed}',
                        'General/validation_split': '${pipeline.validation_split}'} # type: ignore
)

with open('models.txt', 'r') as file:
    model_names = file.read().splitlines()

for model_name in model_names:

    pipe.add_parameter('pre_trained_model', model_name)

    pipe.add_step(
        name = f'{model_name}',
        base_task_name='train_model',
        base_task_project=PROJECT_NAME,
        parents=['data_split'],
        parameter_override={'General/seed': '${pipeline.seed}',
                            'General/pre_trained_model': '${pipeline.pre_trained_model}',
                            'General/batch_size': '${pipeline.batch_size}',
                            'General/max_length': '${pipeline.max_length}',
                            'General/lr': '${pipeline.lr}',
                            'General/stratified_sampling': '${pipeline.stratified_sampling}',
                            'General/stratified_epochs': '${pipeline.stratified_epochs}',
                            'General/num_epochs': '${pipeline.num_epochs}',
                            'General/accelerator': '${pipeline.accelerator}',
                            'General/devices': '${pipeline.devices}'} # type: ignore
    )

if __name__ == '__main__':
    pipe.start()

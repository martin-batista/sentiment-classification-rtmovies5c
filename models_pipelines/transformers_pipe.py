from clearml import Task
from clearml.automation import PipelineController
from pipe_conf import PROJECT_NAME

parameters = {
    'validation_split': 0.1,
    'seed': 42,
    'pre_trained_model': 'bert-base-uncased',
    'batch_size': 16,
    'max_length': 256,
    'lr': 2e-5,
    'num_epochs': 8,
    'stratified_sampling': True,
    'stratified_sampling_position': 'first',
    'stratified_epochs': 4,
    'lr_schedule': 'warmup_cosine_restarts', # warmup_linear, warmup_constant, warmup_cosine_restarts
    'lr_warmup': 0.1,
    'num_cycles': 4,
    'accelerator': 'auto',
    'devices': 'auto',
}

Task.add_requirements('requirements.txt')
pipe = PipelineController(
    name = 'transformers_custom_pipeline',
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
pipe.add_parameter('stratified_sampling_position', parameters['stratified_sampling_position'])
pipe.add_parameter('stratified_epochs', parameters['stratified_epochs'])
pipe.add_parameter('lr_schedule', parameters['lr_schedule'])
pipe.add_parameter('lr_warmup', parameters['lr_warmup'])
pipe.add_parameter('num_cycles', parameters['num_cycles'])
pipe.add_parameter('accelerator', parameters['accelerator'])
pipe.add_parameter('devices', parameters['devices'])

with open('pooler_models.txt', 'r') as file:
    model_names = file.read().splitlines()

for model_name in model_names:
    pipe.add_parameter('pre_trained_model', model_name)
    pipe.add_step(
        name = f'{model_name}',
        base_task_name='base_train_model',
        base_task_project=PROJECT_NAME,
        parameter_override={'General/seed': '${pipeline.seed}',
                            'General/pre_trained_model': '${pipeline.pre_trained_model}',
                            'General/batch_size': '${pipeline.batch_size}',
                            'General/max_length': '${pipeline.max_length}',
                            'General/lr': '${pipeline.lr}',
                            'General/stratified_sampling': '${pipeline.stratified_sampling}',
                            'General/stratified_epochs': '${pipeline.stratified_epochs}',
                            'General/lr_schedule': '${pipeline.lr_schedule}',
                            'General/lr_warmup': '${pipeline.lr_warmup}',
                            'General/num_cycles': '${pipeline.num_cycles}',
                            'General/num_epochs': '${pipeline.num_epochs}',
                            'General/accelerator': '${pipeline.accelerator}',
                            'General/devices': '${pipeline.devices}'
        }
    )

if __name__ == '__main__':
    pipe.start()

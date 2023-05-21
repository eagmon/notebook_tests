from bigraph_viz import plot_bigraph, plot_flow, pf
from bigraph_viz.dict_utils import schema_keys
from sed2.core import register, ports, annotate, Composite, ProcessRegistry
from sed2.processes import sed_process_registry

schema_keys.extend(['_id', 'config'])
sbml_model_path = 'susceptible_zombie.xml'


def ex1():
    # SED document serialized
    instance1 = {
        'time_start': 0,
        'time_end': 10,
        'num_points': 50,
        'selection_list': ['time', 'S', 'Z'],
        'model_path': sbml_model_path,
        'curves': {
            'Susceptible': {'x': 'time', 'y': 'S'},
            'Zombie': {'x': 'time', 'y': 'Z'}
        },
        'figure1name': '"Figure1"',
        'sbml_model_from_path': {
            '_id': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },
        'plot2d': {
            '_id': 'plot2D',
            'wires': {
                'results': 'results',
                'curves': 'curves',
                'name': 'figure1name',
                'figure': 'figure'
            },
            '_depends_on': ['uniform_time_course'],
        },
        'uniform_time_course': {
            '_id': 'uniform_time_course',
            'wires': {
                'model': 'model_instance',
                'time_start': 'time_start',
                'time_end': 'time_end',
                'num_points': 'num_points',
                'selection_list': 'selection_list',
                'results': 'results',
            },
            '_depends_on': ['sbml_model_from_path'],
        },
    }

    sim_experiment = Composite(
        config=instance1,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)

    # print(pf(sim_experiment.config))
    print(results)


def ex2():
    instance2 = {
        'model_path': sbml_model_path,
        'UTC': '"UTC"',
        'selection_list': ['S', 'Z'],
        'sbml_model_from_path': {
            '_id': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },
        'steady_state_values': {
            '_id': 'steady_state',
            'wires': {
                'model': 'model_instance',
                # 'time_start': 'time_start',
                # 'time_end': 'time_end',
                # 'num_points': 'num_points',
                'selection_list': 'selection_list',
                'results': 'results',
            },
            '_depends_on': ['sbml_model_from_path']
        },
        'report': {
            '_id': 'report',
            'wires': {
                'results': 'results',
                'title': 'UTC'  # this should be optional
            },
            '_depends_on': ['steady_state_values']
        }
    }

    sim_experiment = Composite(
        config=instance2,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)

    # print(pf(sim_experiment.config))
    print(results)


if __name__ == '__main__':
    # ex1()
    ex2()
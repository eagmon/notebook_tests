import inspect
import json
import types
import abc
import numpy as np
from bigraph_viz import plot_bigraph, plot_flow, pf
from bigraph_viz.dict_utils import schema_keys

schema_keys.extend(['_id', 'config'])


"""
Decorators
"""


def register(registry, identifier=None):
    def decorator(func):
        registry.register(func, identifier=identifier)
        return func

    return decorator


def annotate(annotation):
    def decorator(func):
        func.annotation = annotation
        return func

    return decorator


# TODO: ports for functions require input/output, but for processes this isn't required
# TODO assert type are in type_registry
# TODO check that keys match function signature
def ports(ports_schema):
    # assert inputs/outputs and types, give suggestions
    allowed = ['inputs', 'outputs']
    assert all(key in allowed for key in
               ports_schema.keys()), f'{[key for key in ports_schema.keys() if key not in allowed]} not allowed as top-level port keys. Allowed keys include {str(allowed)}'

    def decorator(func):
        func.ports = ports_schema
        return func

    return decorator


"""
Registry
"""
class ProcessRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, process, identifier=None):
        if not identifier:
            identifier = process.__name__
        signature = inspect.signature(process)
        annotation = getattr(process, 'annotation', None)
        ports = getattr(process, 'ports')

        try:
            bases = [base.__name__ for base in process.__bases__]
        except:
            bases = None
        process_type = None
        if isinstance(process, types.FunctionType):
            process_type = 'function'
        elif 'Composite' in bases:
            process_type = 'composite'
        elif 'Process' in bases:
            process_type = 'process'

        # TODO -- assert ports and signature match
        if not annotation:
            raise Exception(f'Process {identifier} requires annotations')
        if not ports:
            raise Exception(f'Process {identifier} requires annotations')

        item = {
            'annotation': annotation,
            'ports': ports,
            'address': process,
            'type': process_type,
        }
        self.registry[identifier] = item

    def access(self, name):
        return self.registry.get(name)

    def get_annotations(self):
        return [v.get('annotation') for k, v in self.registry.items()]

    def activate_process(self, process_name, namespace):
        namespace[process_name] = self.registry[process_name]['address']

    def activate_all(self, namespace):
        """how to add to globals: process_registry.activate_all(globals())"""
        for process_name in self.registry.keys():
            self.activate_process(process_name, namespace)


# initialize a registry
sed_process_registry = ProcessRegistry()


"""
More helper functions
"""

def serialize_instance(wiring):
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type "{obj.__class__.__name__}" is not JSON serializable')

    return json.dumps(wiring, default=convert_numpy)


def deserialize_instance(serialized_wiring):
    if isinstance(serialized_wiring, dict):
        return serialized_wiring

    def convert_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj)
        return obj

    return json.loads(serialized_wiring, object_hook=convert_numpy)


def extract_composite_config(schema):
    config = {k: v for k, v in schema.items() if k not in schema_keys}
    return config


def initialize_process_from_schema(schema, process_registry):
    schema = deserialize_instance(schema)
    assert len(schema) == 1  # only one top-level key
    process_name = next(iter(schema))
    process_id = schema[process_name].get('_id', process_name)
    process_entry = process_registry.access(process_id)
    process = process_entry['address']  # get the process from registry
    if process_entry['type'] == 'function':
        return {process_name: process}
    elif process_entry['type'] == 'process' or process_entry['type'] == 'composite':
        config = extract_composite_config(schema[process_name])
        return {process_name: process(config=config, process_registry=process_registry)}


def get_processes_states_from_schema(schema, process_registry):
    all_annotations = process_registry.get_annotations()
    processes = {}
    states = {}
    for name, value in schema.items():
        if isinstance(value, dict) and value.get('wires'):
            processes[name] = value
        else:
            states[name] = value
    return processes, states


"""
Process and Composite base class
"""


class Process:
    config = {}

    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        self.config = config

    @abc.abstractmethod
    def ports(self):
        return {}

    @abc.abstractmethod
    def update(self, state):
        return {}


class Composite(Process):
    config = {}

    def __init__(self, config, process_registry):
        self.initialize(config, process_registry)

    def initialize(self, config, process_registry):
        self.config = config
        self.process_registry = process_registry
        processes, states = get_processes_states_from_schema(
            self.config, self.process_registry)
        self.states = states
        self.processes = {}

        for process in processes:
            process_schema = {process: self.config[process]}
            initialized_process = initialize_process_from_schema(process_schema, process_registry)
            self.processes.update(initialized_process)

    def process_state(self, process_path):
        return

    def to_json(self):
        return serialize_instance(self.config)

    @abc.abstractmethod
    def run(self):
        return {}



"""
Make example processes and composites
"""


@register(
    identifier='loop',
    registry=sed_process_registry)
@ports({
    'inputs': {
        'trials': 'int'},
    'outputs': {
        'results': 'list'}})
@annotate('sed:range_iterator')
class RangeIterator(Composite):
    def run(self, trials):
        results = []
        for i in range(trials):
            for process in self.processes:
                # TODO -- get the process state
                result = process.update()
        return results


@register(
    identifier='sum',
    registry=sed_process_registry)
@ports({
    'inputs': {'values': 'list[float]'},
    'outputs': {'result': 'float'}})
@annotate('math:add')
def add_list(values):
    if not isinstance(values, list):
        values = [values]
    return sum(values)


@register(
    identifier='add_two',
    registry=sed_process_registry)
@ports({
    'inputs': {'a': 'float', 'b': 'float'},
    'outputs': {'result': 'float'}})
@annotate('add_two')
def add_two(a, b):
    return a + b


def test_instance():
    config1 = {
        'trials': 10,
        'loop': {
            '_id': 'loop',
            '_type': 'sed:range_iterator',
            'wires': {
                'trials': 'trials',
            },
            # 'config': {},
            'value': 0,
            'added': 1,
            'add': {
                '_type': 'add_two',
                '_id': 'add_two',
                'wires': {
                    'a': 'value',
                    'b': 'added',
                    'result': 'value',
                },
            }
        },

    }

    sim_experiment = Composite(
        config=config1,
        process_registry=sed_process_registry)

    sim_experiment.run()

    json_str = sim_experiment.to_json()

    plot_bigraph(config1, out_dir='out', filename='test1')


if __name__ == '__main__':
    test_instance()

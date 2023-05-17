import inspect
import copy
import json
import types
import abc
from abc import ABC
from functools import reduce
import operator

import numpy as np
from bigraph_viz import plot_bigraph, pf, pp
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
    ports = copy.deepcopy(ports_schema.get('inputs', {}))
    ports.update(ports_schema.get('outputs', {}))
    def decorator(func):
        func.input_output_ports = ports_schema
        func.ports = ports
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

        process_class = None
        if isinstance(process, types.FunctionType):
            process_class = 'function'
        elif 'Composite' in bases:
            process_class = 'composite'
        elif 'Process' in bases:
            process_class = 'process'
        process.process_class = process_class  # add process class annotation

        # TODO -- assert ports and signature match
        if not annotation:
            raise Exception(f'Process {identifier} requires annotations')
        if not ports:
            raise Exception(f'Process {identifier} requires annotations')

        item = {
            'annotation': annotation,
            'ports': ports,
            'address': process,
            'class': process_class,
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


def get_value_from_path(dictionary, path):
    # noinspection PyBroadException
    try:
        return reduce(operator.getitem, path, dictionary)
    except Exception:
        return None


def extract_composite_config(schema):
    config = {k: v for k, v in schema.items() if k not in schema_keys}
    return config


def get_processes_states_from_schema(schema, process_registry, path=None):
    schema = copy.deepcopy(schema)
    path = path or ()
    all_annotations = process_registry.get_annotations()

    processes = {}
    states = {}
    for name, value in schema.items():
        next_path = path + (name,)
        if isinstance(value, dict):
            if value.get('wires'):
                # get the process
                process_id = value.pop('_id')
                process_type = value.pop('_type')
                process_wires = value.pop('wires')
                process = process_registry.access(process_id)
                process_ports = process['ports']
                assert process_type in all_annotations, f'{name} needs a type annotation from: {all_annotations}'
                assert process_wires.keys() == process_ports.keys(), f'{name} wires {list(process_wires.keys())} ' \
                                                                     f'need to match ports {list(process_ports.keys())}'
                # initialize the process
                process_class = process['class']
                process_address = process['address']
                if process_class == 'function':
                    processes[next_path] = process_address
                elif process_class == 'process':
                    processes[next_path] = process_address(value)
                elif process_class == 'composite':
                    processes[next_path] = process_address(value, process_registry)  # TODO -- get process config, not full value
                    value = {}

            p, s = get_processes_states_from_schema(value, process_registry, path=next_path)
            processes.update(p)
            states.update(s)
        elif name not in schema_keys:
            states[name] = value

    return processes, states


"""
Process and Composite base class
"""


class Process:
    config = {}

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def ports(self):
        return {}

    @abc.abstractmethod
    def update(self, state):
        return {}


class Composite(Process, ABC):
    config = {}
    processes = None
    states = None

    def __init__(self, config, process_registry):
        self.config = config
        self.process_registry = process_registry
        processes, states = get_processes_states_from_schema(
            self.config, self.process_registry)
        self.states = states
        self.processes = processes

    def process_state(self, process_path):
        # TODO -- get the states for this specific process
        process_value = get_value_from_path(self.config, process_path)
        wires = process_value['wires']
        states = {wire_id: self.states[target] for wire_id, target in wires.items()}
        return states

    def to_json(self):
        return serialize_instance(self.config)

    def update(self, state):
        updates = []
        for process_path, process in self.processes.items():
            process_states = self.process_state(process_path)
            if process.process_class == 'function':
                result = process(**process_states)
            else:
                result = process.update(state=process_states)
            updates.append(result)

        return updates


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
@annotate('sed:composite:range_iterator')
class RangeIterator(Composite):
    def update(self, state):
        trials = state.get('trials', 0)
        results = []
        for i in range(trials):
            for process_path, process in self.processes.items():
                # TODO -- get the process state
                process_states = self.process_state(process_path)
                if process.process_class == 'function':
                    input_states = {k: process_states[k] for k in process.input_output_ports['inputs'].keys()}
                    result = process(**input_states)
                else:
                    result = process.update(process_states)
                results.append(result)
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


def run_instance1():


    config1 = {
        'trials': 10,
        'results': None,  # this should be filled in automatically
        'for_loop': {
            '_id': 'loop',
            '_type': 'sed:composite:range_iterator',
            'wires': {
                'trials': 'trials',
                'results': 'results',
            },
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

    state = {}
    results = sim_experiment.update(state=state)

    # print(pf(sim_experiment.config))
    print(results)

    plot_bigraph(config1, out_dir='../composites/out', filename='test1')


if __name__ == '__main__':
    run_instance1()

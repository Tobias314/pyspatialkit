from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union, List
import multiprocessing
import inspect
import importlib


from ..tiling.abstracttiler import AbstractTiler
from ..storage.geostorage import GeoStorage
from ..storage.geolayer import GeoLayer

class LayerProcessor:

    def __init__(self, tiler: AbstractTiler, processor_function: Callable, num_workers: Optional[int] = None):
        self.tiler = tiler
        self.processor_function = processor_function
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        spec = inspect.getfullargspec(self.processor_function)
        self.f_args, self.f_varargs, self.f_varkw, self.f_defaults, self.f_kwonlyargs, self.f_kwkonlydefaults, self.f_annotations = spec
        self.f_args_set = set(self.f_args)
        self.f_kwonlyargs_set = set(self.f_kwonlyargs)

    def __call__(self, *args, **kwargs):
        res_args = []
        res_arg_names = set()
        res_kwargs = {}
        if self.f_varargs is None:
            max_num_positional_args = len(self.f_args)
        else:
            max_num_positional_args = len(args)
        for i, arg in enumerate(args):
            if i >= max_num_positional_args:
                raise AttributeError("Error while preparing layer processing. Too many positional args for fuction {}".format(
                    self.processor_function.__name__))
            res_args.append(arg)
            res_arg_names.add(self.f_args[i])
        for key, arg in kwargs.items():
            if self.f_varkw is False and key not in self.f_args_set and key not in self.f_kwonlyargs_set:
                raise AttributeError(
                    "Error while preparing layer processing. Argument {} not in list of arguments".format(key))
            if key in res_arg_names:
                raise AttributeError("Error while preparing layer processing. Argument {} set twice".format(key))
            res_kwargs[key] = arg
        if self.f_defaults is not None:
            for key, default in zip(self.f_args[len(self.f_args)-len(self.f_defaults):], self.f_defaults):
                if key not in res_kwargs and key not in res_arg_names:
                    res_kwargs[key] = default
            for key, default in self.f_kwkonlydefaults.items():
                if key not in res_kwargs:
                    res_kwargs[key] = default
        workloads = []
        tiler_partitions = self.tiler.partition(self.num_workers)
        for tiler_partition in tiler_partitions:
            workloads.append(LayerProcessorWorkload(tiler_partition, self, res_args, res_kwargs))
        asyncs = []
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=self.num_workers) as pool:
            for workload in workloads:
                asyncs.append(pool.apply_async(process_layer_workload, (workload,)))
            for async_res in asyncs:
                async_res.get()

class LayerProcessorConfigurator:

    def __init__(self, processor_function: Callable):
        self.processor_function = processor_function

    def __call__(self, tiler: AbstractTiler, num_workers: Optional[int] = None) -> LayerProcessor:
        return LayerProcessor(tiler=tiler, processor_function=self.processor_function, num_workers=num_workers)


class LayerProcessorWorkload:

    def __init__(self, tiler_partition: AbstractTiler, processor: LayerProcessor, args: List, kwargs: Dict[str, object]):
        self.tiler_partition = tiler_partition
        self.processor = processor
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self) -> Dict[str, object]:
        processor_function = (inspect.getmodule(
            self.processor.processor_function).__name__, self.processor.processor_function.__name__)
        return {'tiler_partition': self.tiler_partition,
                'processor_function': processor_function,
                'args': self.args,
                'kwargs': self.kwargs}

    def __setstate__(self, state: Dict[str, object]):
        self.tiler_partition = state['tiler_partition']
        processor_function = state['processor_function']
        mod = importlib.import_module(processor_function[0])
        self.processor = getattr(mod, processor_function[1])
        self.args = state['args']
        self.kwargs = state['kwargs']


def process_layer_workload(workload: LayerProcessorWorkload):
    for tile in workload.tiler_partition:
        workload.processor.processor_function(tile, *workload.args, **workload.kwargs)


def layerprocessor(func):
    return LayerProcessorConfigurator(processor_function=func)
        

import re

from mosaic import tessera
from mosaic.utils import snake_case

from .steps import steps_registry
from .steps.dump import Dump
from ...core import Operator, no_grad


__all__ = ['Pipeline']


@tessera
class Pipeline(Operator):
    """
    A pipeline represents a series of processing steps that will be applied
    in order to a series of inputs. Pipelines encode pre-processing or
    post-processing steps such as filtering time traces or smoothing a gradient.

    Parameters
    ----------
    steps : list, optional
        List of steps that form the pipeline. Steps can be callable or strings pointing
        to a default, pre-defined step.

    """

    def __init__(self, steps=None, **kwargs):
        super().__init__(**kwargs)

        self._no_grad = kwargs.pop('no_grad', False)
        self._kwargs = kwargs

        steps = steps or []

        cls_name = snake_case(self.__class__.__name__)
        dump_re = re.compile(r'^dump_(before|after)_(\S+)$')
        for k, v in kwargs.items():
            match = dump_re.match(k)
            if dump_re.match(k) and v is True:
                pos = match.group(1)
                step = match.group(2)

                if step == cls_name:
                    if pos == 'before':
                        idx = 0
                    else:
                        idx = len(steps)-1
                elif step in steps:
                    idx = steps.index(step)
                elif (step, False) in steps:
                    idx = steps.index((step, False))
                else:
                    continue

                if pos == 'before':
                    steps.insert(idx, 'dump')
                else:
                    steps.insert(idx+1, 'dump')

        self._steps = []
        for step in steps:
            do_raise = True
            if isinstance(step, tuple):
                step, do_raise = step

            if isinstance(step, str):
                step_cls = steps_registry.get(step, None)
                if step_cls is None and do_raise:
                    raise ValueError('Pipeline step %s does not exist in the registry' % step)

                if step_cls is not None:
                    self._steps.append(step_cls(**kwargs))
            else:
                self._steps.append(step)

    async def forward(self, *args, **kwargs):
        """
        Apply all steps in the pipeline in order.

        """
        next_args = args

        prev_step = None
        for step in self._steps:
            if self._no_grad:
                with no_grad(*next_args, **kwargs):
                    next_args = await step(*next_args, **{**self._kwargs, **kwargs},
                                           prev_step=prev_step)
            else:
                next_args = await step(*next_args, **{**self._kwargs, **kwargs},
                                       prev_step=prev_step)
            next_args = (next_args,) if len(args) == 1 else next_args
            prev_step = None if isinstance(step, Dump) else step

        if len(args) == 1:
            return next_args[0]

        else:
            return next_args

    async def adjoint(self, *args, **kwargs):
        input_args, input_kwargs = self.inputs

        outputs = args[:self.num_outputs]

        prev_step = None
        for step in reversed(self._steps):
            outputs = step.adjoint(*outputs, *input_args, **{**self._kwargs, **kwargs},
                                   prev_step=prev_step)
            prev_step = None if isinstance(step, Dump) else step

        if len(outputs) == 1:
            return outputs[0]

        else:
            return outputs

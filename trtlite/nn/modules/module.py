import contextlib
import threading
from collections import OrderedDict
from typing import Any, Callable, Iterator, Optional, Set, Tuple, TypeVar

from ...utils import hooks
from ...utils.hooks import RemovableHandle
from .context import DefaultTrtContext, TrtContext

T = TypeVar('T', bound='Module')


class NameScopeStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super(NameScopeStack, self).__init__()
    self.stack = []
    
  def reset(self):
    self.stack = []

  def is_cleared(self):
    return not self.stack

  def name(self):
    return self.stack[-1] if self.stack else ''

  @contextlib.contextmanager
  def scope_name(self, name):
    """A context manager for manipulating a default stack."""
    self.stack.append(name)
    try:
      yield name
    finally:
      # stack may be empty if reset() was called
      if self.stack:
        self.stack.pop()


_DEFAULT_NAME_SCOPE_STACK = NameScopeStack()


def module_with_name_scope(module, module_name):

    def forward(*input, **kwargs):
        name = get_current_module_name()
        if len(name) > 0:
            new_name = f"{name}.{module_name}" 
        else: 
            new_name = module_name

        with get_default_scope_name(new_name):
            x = ori_forward(*input, **kwargs)
        
        return x

    ori_forward = module.forward
    module.forward = None
    module.forward = forward
    return module


def get_current_module_name():
    return _DEFAULT_NAME_SCOPE_STACK.name()


def get_default_scope_name(name):
    return _DEFAULT_NAME_SCOPE_STACK.scope_name(name)


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def _forward_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.
    """
    raise NotImplementedError


class Module:
    r"""Base class for all neural network modules.
    """

    _version: int = 1

    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        self._modules = OrderedDict()
        self._forward_hooks = OrderedDict()

    forward: Callable[..., Any] = _forward_unimplemented

    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {name}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module_with_name_scope(module, name)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _forward(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def _call_impl(self, *input, **kwargs):
        forward_call = self._forward
        if not self._forward_hooks:
            return forward_call(*input, **kwargs)

        result = forward_call(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        return result


    __call__ : Callable[..., Any] = _call_impl


    def __getattr__(self, name: str) -> Optional['Module']:
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Optional['Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            remove_from(self.__dict__)
            modules[name] = module_with_name_scope(value, name)
            # create full module name for each module

        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(f"cannot assign '{value}' "
                                f"as child module '{name}' "
                                "(nn.Module or None expected)")
            modules[name] = value
        else:
            object.__setattr__(self, name, value)
       

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)


    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v


    def register_forward_hook(self,
                              hook: Callable[..., None]) -> RemovableHandle:
        r"""Registers a forward hook on the module."""
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle


    def children(self) -> Iterator['Module']:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        r"""Returns an iterator over all modules in the network.
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, 
                      memo: Optional[Set['Module']] = None, 
                      prefix: str = ''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple of name and module
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + modules
        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def build_engine(self, config):
        engine_file = config['engine_file']
        weights_file = config['weight_file']
        inputs = config['inputs']
        input_profiles = config['input_profiles']
        max_workspace_size = config.get('max_workspace_size', 1 << 30)
        precision = config.get('precision', 'fp32')
        output_names = config.get('output_names', None)

        ctx = TrtContext(weights_map_file=weights_file, 
                         max_workspace_size=max_workspace_size,
                         precision=precision)
        ctx.initialize()
        input_names = [name for name, _ in inputs]
        input_shapes = [shape for _, shape in inputs]

        with DefaultTrtContext(ctx) as ctx:
            xs = ctx.add_inputs(input_names, input_shapes)
            outputs = self(*xs)
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
            
            ctx.mark_outputs(outputs, output_names)
   
        ctx.set_optimization_profiles(input_names, input_profiles)
        ctx.build_engine(engine_file)

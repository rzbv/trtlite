class Rigistry:
    def __init__(self, name=''):
        self._name = name
        self._func_dict = dict()

    def __len__(self):
        return len(self._func_dict)

    @property
    def name(self):
        return self._name

    @property
    def func_dict(self):
        return self._func_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        return self._func_dict.get(key, None)

    def register(self, op_name, version=None):
        def decorator(func):
            """Register a function as a plug-in."""
            if self.get(op_name):
                if version in self.get(op_name):
                    raise KeyError(f'{version} is already registered '
                                   f'in {op_name}')
                self._func_dict[op_name][version] = (func.__name__, func)
            else:
                self._func_dict[op_name] = {version: (func.__name__, func)}

            return func

        # decorator.all = tmp
        return decorator

# The most functions are mainly referenced from torch.utils.cpp_extention
# modified logics: 1. remove torch dependence; 2. only use for environment
# ==============================================================================

import copy
import os
import subprocess
import sys
import warnings
from typing import List, Optional

import setuptools

# from pkg_resources import packaging  # type: ignore
from setuptools.command.build_ext import build_ext

IS_WINDOWS = sys.platform == 'win32'
MINIMUM_GCC_VERSION = (5, 0, 0)
ABI_INCOMPATIBILITY_WARNING = '{} CONFLICT WITH ABI_INCOMPATIBILITY_WARNING'


def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(
                    [which, 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None

    return cuda_home


def _get_cuda_version(cuda_home: str) -> Optional[str]:
    version_file = os.path.join(cuda_home, 'version.txt')
    ver = None
    if os.path.exists(version_file):
        content = open('version_file').read()
        ver = content.strip().rsplit('.', 1)[0]

    return ver


CUDA_HOME = _find_cuda_home()
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
COMMON_NVCC_FLAGS = []


def check_compiler_abi_compatibility(compiler) -> bool:
    r'''
    Verifies that the given compiler is ABI-compatible with PyTorch.
    '''
    minimum_required_version = MINIMUM_GCC_VERSION
    versionstr = subprocess.check_output(
        [compiler, '-dumpfullversion', '-dumpversion'])
    version = versionstr.decode().strip().split('.')
    if tuple(map(int, version)) >= minimum_required_version:
        return True

    compiler = f'{compiler} {".".join(version)}'
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
    return False


class BuildExtension(build_ext, object):
    r'''
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++14``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``nvcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.
    '''
    @classmethod
    def with_options(cls, **options):
        r'''
        Returns a subclass with alternative constructor that
        extends any original keyword arguments to the original
        constructor with the given options.
        '''
        class cls_with_options(cls):  # type: ignore
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get('no_python_abi_suffix', False)

    def finalize_options(self) -> None:
        super().finalize_options()

    def build_extensions(self) -> None:
        self._check_abi()
        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'nvcc' is
            # passed to extra_compile_args in CUDAExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            # See note [Pybind11 ABI constants]
            # params = [("COMPILER_TYPE", '_cxxabi1011'), ('STDLIB', '_gcc'),
            #           ('BUILD_ABI', '_libstdcpp')]
            params = [('STDLIB', '_gcc'), ('BUILD_ABI', '_libstdcpp')]
            for name, val in params:
                if val is not None:
                    self._add_compile_flag(extension,
                                           f'-DPYBIND11_{name}="{val}"')

        # Register .cu, .cuh and .hip as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh']
        # Save the original _compile method for later.
        original_compile = self.compiler._compile

        def append_std14_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++14'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (COMMON_NVCC_FLAGS + ['--compiler-options', "'-fPIC'"] +
                      cflags + _get_cuda_arch_flags(cflags))

            # NVCC does not allow multiple -ccbin/--compiler-bindir
            # to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv('CC')
            if (_ccbin is not None and not any([
                    f.startswith('-ccbin') or f.startswith('--compiler-bindir')
                    for f in cflags
            ])):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs,
                                     pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [_join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']

                append_std14_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        self.compiler._compile = unix_wrap_single_compile
        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super(BuildExtension, self).get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension",
            # "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def _check_abi(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif IS_WINDOWS:
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')
        check_compiler_abi_compatibility(compiler)

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(
            extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        pass


def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
                name='extension',
                ext_modules=[
                    CppExtension(
                        name='extension',
                        sources=['extension.cpp'],
                        extra_compile_args=['-g']),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    # libraries.append('c10')
    # libraries.append('torch')
    # libraries.append('torch_cpu')
    # libraries.append('torch_python')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
                name='cuda_extension',
                ext_modules=[
                    CUDAExtension(
                            name='cuda_extension',
                            sources=['extension.cpp', 'extension_kernel.cu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'nvcc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })

    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    # libraries.append('c10')
    # libraries.append('torch')
    # libraries.append('torch_cpu')
    # libraries.append('torch_python')

    libraries.append('cudart')

    # libraries.append('c10_cuda')
    # if BUILD_SPLIT_CUDA:
    #     libraries.append('torch_cuda_cu')
    #     libraries.append('torch_cuda_cpp')
    # else:
    #     libraries.append('torch_cuda')

    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs
    kwargs['language'] = 'c++'

    return setuptools.Extension(name, sources, *args, **kwargs)


def include_paths(cuda: bool = False) -> List[str]:
    """Get the include paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific include paths.
    """

    paths = []
    if cuda:
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda,
        # we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    return paths


def library_paths(cuda: bool = False) -> List[str]:
    paths = []
    if cuda:
        lib_dir = 'lib64'
        if (not os.path.exists(_join_cuda_home(lib_dir))
                and os.path.exists(_join_cuda_home('lib'))):
            # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
            # Note that it's also possible both don't exist (see
            # _find_cuda_home) - in that case we stay with 'lib64'.
            lib_dir = 'lib'

        paths.append(_join_cuda_home(lib_dir))
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, lib_dir))
    return paths


def _get_cuda_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    r'''
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    '''
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if 'arch' in flag:
                return []

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    # named_arches = collections.OrderedDict([
    #     ('Kepler+Tesla', '3.7'),
    #     ('Kepler', '3.5+PTX'),
    #     ('Maxwell+Tegra', '5.3'),
    #     ('Maxwell', '5.0;5.2+PTX'),
    #     ('Pascal', '6.0;6.1+PTX'),
    #     ('Volta', '7.0+PTX'),
    #     ('Turing', '7.5+PTX'),
    #     ('Ampere', '8.0;8.6+PTX'),# 3060,3080/3090
    # ])

    supported_arches = [
        '5.3', '6.0', '6.1', '6.2', '7.0', '7.2', '7.5', '8.0', '8.6+PTX'
    ]
    # valid_arch_strings = supported_arches

    cuda_ver = _get_cuda_version(CUDA_HOME)
    if '8.0' in supported_arches or '8.6+PTX' in supported_arches:
        cuda_ver = int(cuda_ver.split('.')[0])
        assert cuda_ver >= 11, 'arch=8.0+ need cuda 11+'

    flags = []
    for arch in supported_arches:
        num = arch.split('+')[0].replace('.', '')
        flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
        if arch.endswith('+PTX'):
            flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    return sorted(list(set(flags)))


def _join_cuda_home(*paths) -> str:
    r'''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    return os.path.splitext(path)[1] in valid_ext

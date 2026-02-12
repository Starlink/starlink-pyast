import ctypes
import concurrent.futures
import importlib.util
import os
import sys
import tarfile
from textwrap import dedent

import numpy
from setuptools import Extension, setup
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler
from setuptools.command.build_ext import build_ext

cwd = os.path.abspath(os.path.dirname(__file__))


def _find_tools_dir():
    # When running the build from setuptools in a sandbox we can not
    # directly import the helper code. Instead we need to find it relative
    # to the setup.py file.
    candidates = (
        cwd,
        os.getcwd(),
    )
    for base in candidates:
        tools_dir = os.path.join(base, "tools")
        if os.path.isfile(os.path.join(tools_dir, "make_attributes.py")):
            return tools_dir
    for base in candidates:
        current = os.path.abspath(base)
        while True:
            tools_dir = os.path.join(current, "tools")
            if os.path.isfile(os.path.join(tools_dir, "make_attributes.py")):
                return tools_dir
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    raise RuntimeError(
        "Unable to locate build support modules in tools/. "
        "Expected tools/make_attributes.py and tools/make_exceptions.py."
    )


def _load_build_tool(module_name, tools_dir):
    # In the setuptools sandbox we cannot directly import from the tools
    # directly and so instead have to use the importlib APIs to load the
    # file directly.
    module_file = os.path.join(tools_dir, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load build support module: {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tools_dir = _find_tools_dir()
make_attributes = _load_build_tool("make_attributes", _tools_dir)
make_exceptions = _load_build_tool("make_exceptions", _tools_dir)


def get_compiler():
    """Get the compiler.

    Use the compiler APIs provided by setuptools' distutils shim.
    """
    compiler = new_compiler()
    customize_compiler(compiler)
    return compiler


class BuildExt(build_ext):
    """Use all CPUs by default and parallelize C source compilation."""

    def finalize_options(self):
        super().finalize_options()
        if self.parallel is None:
            self.parallel = os.cpu_count() or 1

    def build_extensions(self):
        jobs = self.parallel or 1
        compiler = self.compiler

        # build_ext.parallel only parallelizes across extensions; this project
        # has one large extension, so compile source files in parallel instead.
        if jobs <= 1 or not all(hasattr(compiler, name) for name in ("_setup_compile", "_get_cc_args", "_compile")):
            return super().build_extensions()

        original_compile = compiler.compile

        def parallel_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            macros, objects, extra_postargs, pp_opts, build = compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            cc_args = compiler._get_cc_args(pp_opts, debug, extra_preargs)

            def _compile_one(obj):
                src, ext = build[obj]
                compiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(objects) or 1)) as executor:
                list(executor.map(_compile_one, objects))

            return objects

        compiler.compile = parallel_compile
        saved_parallel = self.parallel
        self.parallel = None
        try:
            super().build_extensions()
        finally:
            compiler.compile = original_compile
            self.parallel = saved_parallel


def check_libyaml():
    """check if the C module can be build by trying to compile a small
    program against the libyaml development library"""

    import shutil
    import tempfile

    libraries = ["yaml"]

    # write a temporary .c file to compile
    c_code = dedent(
        """
    #include <yaml.h>

    int main(int argc, char* argv[])
    {
        yaml_parser_t parser;
        parser = parser;  /* prevent warning */
        return 0;
    }
    """
    )
    tmp_dir = tempfile.mkdtemp(prefix="tmp_ruamel_yaml_")
    bin_file_name = os.path.join(tmp_dir, "test_yaml")
    file_name = bin_file_name + ".c"
    with open(file_name, "w") as fp:
        fp.write(c_code)

    # and try to compile it
    compiler = get_compiler()

    try:
        compiler.link_executable(
            compiler.compile([file_name]),
            bin_file_name,
            libraries=libraries,
        )
    except Exception as e:
        print(f"libyaml compilation error: {e}")
        ret_val = False
    else:
        ret_val = True

    shutil.rmtree(tmp_dir)
    return ret_val


include_dirs = []

include_dirs.append(numpy.get_include())
include_dirs.append(os.path.join(".", "src", "starlink", "include"))
include_dirs.append(os.path.join(".", "ast"))
include_dirs.append(os.path.join(".", "ast", "src"))

#  Create the support files needed for the build. These find the AST
#  source code using the environment variable AST_SOURCE, so set AST_SOURCE
#  to point to the AST source code directory distributed with PyAST.
os.environ["AST_SOURCE"] = os.path.join(cwd, "ast")
make_exceptions.make_exceptions(os.path.join("src", "starlink", "ast"))
make_attributes.make_attributes(os.path.join("src", "starlink", "ast"))

#  Extract the AST documentation
if not os.path.exists("sun211.htx"):
    tar = tarfile.open("ast/sun211.htx_tar")
    tar.extractall()
    tar.close()

#  List the cminpack source files required by AST:
cminpack_c = ("enorm.c", "lmder.c", "qrfac.c", "dpmpar.c", "lmder1.c", "lmpar.c", "qrsolv.c")

#  List the C source files for implemented AST classes (in ast/src
#  subdirectory):
ast_c = (
    "axis.c",
    "box.c",
    "channel.c",
    "circle.c",
    "cmpframe.c",
    "cmpmap.c",
    "cmpregion.c",
    "dsbspecframe.c",
    "dssmap.c",
    "ellipse.c",
    "error.c",
    "fitschan.c",
    "fluxframe.c",
    "frame.c",
    "frameset.c",
    "globals.c",
    "grf3d.c",
    "grf_2.0.c",
    "grf_3.2.c",
    "grf_5.6.c",
    "grismmap.c",
    "interval.c",
    "keymap.c",
    "loader.c",
    "lutmap.c",
    "mapping.c",
    "mathmap.c",
    "matrixmap.c",
    "memory.c",
    "moc.c",
    "mocchan.c",
    "normmap.c",
    "nullregion.c",
    "object.c",
    "pcdmap.c",
    "permmap.c",
    "plot.c",
    "pointlist.c",
    "pointset.c",
    "polygon.c",
    "polymap.c",
    "prism.c",
    "ratemap.c",
    "region.c",
    "shiftmap.c",
    "skyaxis.c",
    "skyframe.c",
    "specfluxframe.c",
    "specframe.c",
    "sphmap.c",
    "stcschan.c",
    "timeframe.c",
    "timemap.c",
    "tranmap.c",
    "unit.c",
    "unitmap.c",
    "wcsmap.c",
    "winmap.c",
    "xml.c",
    "xphmap.c",
    "zoommap.c",
    "specmap.c",
    "splinemap.c",
    "slamap.c",
    "chebymap.c",
    "unitnormmap.c",
    "yamlchan.c",
)

#  List the other required C source files (in ast subdirectory):
ast_c2 = ("palwrap.c", "pyast_extra.c")

#  List the other required C source files (in ast/wcslib subdirectory):
ast_c3 = ("proj.c", "tpn.c", "wcstrig.c")

#  List the erfa source files required by AST.
erfa_c = (
    "a2af.c",
    "a2tf.c",
    "ab.c",
    "af2a.c",
    "anp.c",
    "anpm.c",
    "apcg.c",
    "apcg13.c",
    "apci.c",
    "apci13.c",
    "apco.c",
    "apco13.c",
    "apcs.c",
    "apcs13.c",
    "aper.c",
    "aper13.c",
    "apio.c",
    "apio13.c",
    "atci13.c",
    "atciq.c",
    "atciqn.c",
    "atciqz.c",
    "atco13.c",
    "atic13.c",
    "aticq.c",
    "aticqn.c",
    "atio13.c",
    "atioq.c",
    "atoc13.c",
    "atoi13.c",
    "atoiq.c",
    "bi00.c",
    "bp00.c",
    "bp06.c",
    "bpn2xy.c",
    "c2i00a.c",
    "c2i00b.c",
    "c2i06a.c",
    "c2ibpn.c",
    "c2ixy.c",
    "c2ixys.c",
    "c2s.c",
    "c2t00a.c",
    "c2t00b.c",
    "c2t06a.c",
    "c2tcio.c",
    "c2teqx.c",
    "c2tpe.c",
    "c2txy.c",
    "cal2jd.c",
    "cp.c",
    "cpv.c",
    "cr.c",
    "d2dtf.c",
    "d2tf.c",
    "dat.c",
    "dtdb.c",
    "dtf2d.c",
    "eceq06.c",
    "ecm06.c",
    "ee00.c",
    "ee00a.c",
    "ee00b.c",
    "ee06a.c",
    "eect00.c",
    "eform.c",
    "eo06a.c",
    "eors.c",
    "epb.c",
    "epb2jd.c",
    "epj.c",
    "epj2jd.c",
    "epv00.c",
    "eqec06.c",
    "eqeq94.c",
    "era00.c",
    "fad03.c",
    "fae03.c",
    "faf03.c",
    "faju03.c",
    "fal03.c",
    "falp03.c",
    "fama03.c",
    "fame03.c",
    "fane03.c",
    "faom03.c",
    "fapa03.c",
    "fasa03.c",
    "faur03.c",
    "fave03.c",
    "fk52h.c",
    "fk5hip.c",
    "fk5hz.c",
    "fw2m.c",
    "fw2xy.c",
    "g2icrs.c",
    "gc2gd.c",
    "gc2gde.c",
    "gd2gc.c",
    "gd2gce.c",
    "gmst00.c",
    "gmst06.c",
    "gmst82.c",
    "gst00a.c",
    "gst00b.c",
    "gst06.c",
    "gst06a.c",
    "gst94.c",
    "h2fk5.c",
    "hfk5z.c",
    "icrs2g.c",
    "ir.c",
    "jd2cal.c",
    "jdcalf.c",
    "ld.c",
    "ldn.c",
    "ldsun.c",
    "lteceq.c",
    "ltecm.c",
    "lteqec.c",
    "ltp.c",
    "ltpb.c",
    "ltpecl.c",
    "ltpequ.c",
    "num00a.c",
    "num00b.c",
    "num06a.c",
    "numat.c",
    "nut00a.c",
    "nut00b.c",
    "nut06a.c",
    "nut80.c",
    "nutm80.c",
    "obl06.c",
    "obl80.c",
    "p06e.c",
    "p2pv.c",
    "p2s.c",
    "pap.c",
    "pas.c",
    "pb06.c",
    "pdp.c",
    "pfw06.c",
    "plan94.c",
    "pm.c",
    "pmat00.c",
    "pmat06.c",
    "pmat76.c",
    "pmp.c",
    "pmpx.c",
    "pmsafe.c",
    "pn.c",
    "pn00.c",
    "pn00a.c",
    "pn00b.c",
    "pn06.c",
    "pn06a.c",
    "pnm00a.c",
    "pnm00b.c",
    "pnm06a.c",
    "pnm80.c",
    "pom00.c",
    "ppp.c",
    "ppsp.c",
    "pr00.c",
    "prec76.c",
    "pv2p.c",
    "pv2s.c",
    "pvdpv.c",
    "pvm.c",
    "pvmpv.c",
    "pvppv.c",
    "pvstar.c",
    "pvtob.c",
    "pvu.c",
    "pvup.c",
    "pvxpv.c",
    "pxp.c",
    "refco.c",
    "rm2v.c",
    "rv2m.c",
    "rx.c",
    "rxp.c",
    "rxpv.c",
    "rxr.c",
    "ry.c",
    "rz.c",
    "s00.c",
    "s00a.c",
    "s00b.c",
    "s06.c",
    "s06a.c",
    "s2c.c",
    "s2p.c",
    "s2pv.c",
    "s2xpv.c",
    "sepp.c",
    "seps.c",
    "sp00.c",
    "starpm.c",
    "starpv.c",
    "sxp.c",
    "sxpv.c",
    "t_erfa_c.c",
    "taitt.c",
    "taiut1.c",
    "taiutc.c",
    "tcbtdb.c",
    "tcgtt.c",
    "tdbtcb.c",
    "tdbtt.c",
    "tf2a.c",
    "tf2d.c",
    "tr.c",
    "trxp.c",
    "trxpv.c",
    "tttai.c",
    "tttcg.c",
    "tttdb.c",
    "ttut1.c",
    "ut1tai.c",
    "ut1tt.c",
    "ut1utc.c",
    "utctai.c",
    "utcut1.c",
    "xy06.c",
    "xys00a.c",
    "xys00b.c",
    "xys06a.c",
    "zp.c",
    "zpv.c",
    "zr.c",
)

#  List the C source files for unimplemeneted AST classes in ast/src:
ast_c_extra = (
    "fitstable.c",
    "intramap.c",
    "plot3d.c",
    "selectormap.c",
    "stc.c",
    "stccatalogentrylocation.c",
    "stcobsdatalocation.c",
    "stcresourceprofile.c",
    "stcsearchlocation.c",
    "switchmap.c",
    "table.c",
    "xmlchan.c",
)

#  Initialise the list of sources files needed to build the starlink.Ast
#  module.
sources = [os.path.join("src", "starlink", "ast", "Ast.c")]

#  Append all the .c and .h files needed to build the AST library locally.
for cfile in ast_c:
    sources.append(os.path.join("ast", "src", cfile))
for cfile in ast_c2:
    sources.append(os.path.join("ast", cfile))
for cfile in ast_c3:
    sources.append(os.path.join("ast", "wcslib", cfile))
for cfile in cminpack_c:
    sources.append(os.path.join(os.path.join("ast", "cminpack"), cfile))
for cfile in erfa_c:
    sources.append(os.path.join(os.path.join("ast", "erfa"), cfile))
for cfile in ast_c_extra:
    sources.append(os.path.join("ast", "src", cfile))

extra_link_args = []

# Test the compiler
define_macros = []
compiler = get_compiler()
if compiler.has_function("strtok_r"):
    define_macros.append(("HAVE_STRTOK_R", "1"))

if compiler.has_function("strerror_r"):
    define_macros.append(("HAVE_STRERROR_R", "1"))

if check_libyaml():
    define_macros.append(("YAML", "1"))
    extra_link_args.append("-lyaml")

#  We need to tell AST what type a 64-bit int will have
#  Not really sure how to determine whether we have int64_t

define_macros.append(("SIZEOF_LONG", ctypes.sizeof(ctypes.c_long)))
define_macros.append(("SIZEOF_LONG_LONG", ctypes.sizeof(ctypes.c_longlong)))

# isfinite is from C99 so we can now assume this exists.
define_macros.append(("HAVE_DECL_ISFINITE", "1"))

# Assume we have isnan() available and assume we have a working sscanf
# configure would test for these but we no longer run configure
define_macros.append(("HAVE_DECL_ISNAN", "1"))

# Modern compilers are going to have INT64_T
define_macros.append(("HAVE_INT64_T", "1"))
define_macros.append(("HAVE_UINT64_T", "1"))

#  Create the description of the starlink.Ast module.
Ast = Extension("starlink.Ast", include_dirs=include_dirs, define_macros=define_macros, sources=sources)

# OSX needs to hide all the normal AST symbols to prevent
# name clashes when loaded alongside libast itself (eg from pyndf)
symbol_list = "public_symbols.txt"
if sys.platform.startswith("darwin"):
    with open(symbol_list, "w") as symfile:
        print("_PyInit_Ast", file=symfile)
    extra_link_args.append("-exported_symbols_list")
    extra_link_args.append(symbol_list)


if len(extra_link_args) > 0:
    Ast.extra_link_args = extra_link_args

setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[Ast],
    py_modules=["starlink.Grf", "starlink.Atl"],
)

if os.path.exists(symbol_list):
    os.unlink(symbol_list)

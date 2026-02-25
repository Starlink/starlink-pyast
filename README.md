PyAST is a Python extension that provides an interface to the Starlink
AST library. It requires Python v2.7 or later. It can be obtained from
<http://pypi.python.org/pypi/starlink-pyast/>. To install, do:

    $ pip install starlink-pyast

or when building locally:

    $ pip install .
    $ python -m build
    $ python setup.py install --prefix=<installation directory>

To test it, do:

    $ python src/starlink/ast/test/test.py

User docs are available at http://starlink.github.io/starlink-pyast/pyast.html


## History

### 4.0.1

* Remove GPL license file from AST subdirectory that was included by mistake.
  The package has always used LGPL and this change removes some confusion.

### 4.0.0

* Update AST to version 9.3.0.
* New minimum version of Python 3.11.
* Support Python 3.14 and numpy 2.0.
* Add support for SplineMap mapping.
* Fixed handling of `options` parameters in constructors (they were always ignored previously).
* Many internal cleanups associated with no longer having to support legacy versions.
* Improved detection of YAML library in a Conda environment.
* Now support standard build tooling and metadata via `pyproject.toml`.

### 3.15.4

* Upgrade AST internals to version 9.2.5.

## Licence

   This program is free software: you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation, either
   version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   License along with this program.  If not, see
   <http://www.gnu.org/licenses/>.

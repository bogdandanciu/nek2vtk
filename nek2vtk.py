from __future__ import annotations
import os
import io
import struct
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
import numpy as np
from attrs import define, field
from pymech.log import logger
import time
import h5py
import json
import meshio

# ---------------------------------- INPUTS -------------------------------------
# Specify source nek fld file name
fnamenek_in = 'test_cycle0.f00001'
# Specifiy reference quantities .json file name
fnamejson_in = 'test_cycle.json'

# Specify vtk output options
write_vtk_file = True
fnamevtk_out = 'test_cycle.vtk'
vtk_file_format = 'vtk42'  # vtk, vtk42, vtu
# Specify hdf5 output options
write_hdf5_file = False
fnamehdf5_out = 'test_cycle.h5'
# -------------------------------------------------------------------------------

t00 = time.time()


def _as_unicode(string: StringOrBytes) -> str:
    if isinstance(string, bytes):
        return string.decode()
    else:
        return string


def _as_tuple_of_ints(seq):
    return tuple(int(s) for s in seq)


StringOrBytes = Union[str, bytes]
PathLike = Union[str, os.PathLike]


class ElemData:
    """A class containing data related to a hexahedral mesh"""

    def __init__(self, ndim, nel, lr1, var, ncellv, nbc=0, dtype="float64"):
        self.ndim = ndim
        self.nel = nel
        self.ncurv = []
        self.nbc = nbc
        self.var = var
        self.lr1 = lr1
        self.time = []
        self.istep = []
        self.wdsz = []
        self.endian = []
        self.ncellv = ncellv
        self.npel = lr1[2] * lr1[1] * lr1[0]
        self.npts = nel * self.npel
        if lr1[2] > 1:
            self.ncell = nel * (lr1[2]-1) * (lr1[1]-1) * (lr1[0]-1)
        else:
            self.ncell = nel * 1 * (lr1[1] - 1) * (lr1[0] - 1)
        if isinstance(dtype, type):
            # For example np.float64 -> "float64"
            dtype = dtype.__name__
        self.elmap = np.linspace(1, nel, nel, dtype=np.int32)
        #
        self.pos = np.zeros((3, self.npts), dtype=dtype)
        #
        self.vel = np.zeros((3, self.npts), dtype=dtype)
        #
        self.pres = np.zeros((var[2], self.npts), dtype=dtype)
        #
        self.temp = np.zeros((var[3], self.npts), dtype=dtype)
        #
        self.scal = np.zeros((var[4], self.npts), dtype=dtype)
        #
        self.cell_vert = np.zeros((self.ncell, ncellv), dtype=dtype)


@define
class Header:
    """Dataclass for Nek5000 field file header. This relies on the package
    attrs_ and its ability to do type-validation and type-conversion of the
    header metadata.

    .. _attrs: https://www.attrs.org/en/stable/

    """

    # Get word size: single or double precision
    wdsz: int = field(converter=int)
    # Get polynomial order
    orders: Tuple[int, ...] = field(converter=_as_tuple_of_ints)
    # Get number of elements
    nb_elems: int = field(converter=int)
    # Get number of elements in the file
    nb_elems_file: int = field(converter=int)
    # Get current time
    time: float = field(converter=float)
    # Get current time step
    istep: int = field(converter=int)
    # Get file id
    fid: int = field(converter=int)
    # Get tot number of files
    nb_files: int = field(converter=int)

    # NOTE: field(factory=...) specifies the default value for the field.
    # https://www.attrs.org/en/stable/init.html#defaults

    # Get variables [XUPTS[01-99]]
    variables: str = field(converter=_as_unicode, factory=str)
    # Floating point precision
    realtype: str = field(factory=str)
    # Compute total number of points per element
    nb_pts_elem: int = field(factory=int)
    # Get number of physical dimensions
    nb_dims: int = field(factory=int)
    # Get number of variables
    nb_vars: Tuple[int, ...] = field(factory=tuple)

    def __attrs_post_init__(self):
        # Get word size: single or double precision
        wdsz = self.wdsz
        if not self.realtype:
            if wdsz == 4:
                self.realtype = "f"
            elif wdsz == 8:
                self.realtype = "d"
            else:
                logger.error(f"Could not interpret real type (wdsz = {wdsz})")

        orders = self.orders
        if not self.nb_pts_elem:
            self.nb_pts_elem = np.prod(orders)

        if not self.nb_dims:
            self.nb_dims = 2 + int(orders[2] > 1)

        if not self.variables and not self.nb_vars:
            raise ValueError("Both variables and nb_vars cannot be uninitialized.")
        elif self.variables:
            self.nb_vars = self._variables_to_nb_vars()
        elif self.nb_vars:
            self.variables = self._nb_vars_to_variables()

        logger.debug(f"Variables: {self.variables}, nb_vars: {self.nb_vars}")

    def _variables_to_nb_vars(self) -> Optional[Tuple[int, ...]]:
        # get variables [XUPTS[01-99]]
        variables = self.variables
        nb_dims = self.nb_dims

        if not variables:
            logger.error("Failed to convert variables to nb_vars")
            return None

        if not nb_dims:
            logger.error("Unintialized nb_dims")
            return None

        def nb_scalars():
            index_s = variables.index("S")
            return int(variables[index_s + 1 :])

        nb_vars = (
            nb_dims if "X" in variables else 0,
            nb_dims if "U" in variables else 0,
            1 if "P" in variables else 0,
            1 if "T" in variables else 0,
            nb_scalars() if "S" in variables else 0,
        )

        return nb_vars

    def _nb_vars_to_variables(self) -> Optional[str]:
        nb_vars = self.nb_vars
        if not nb_vars:
            logger.error("Failed to convert nb_vars to variables")
            return None

        str_vars = ("X", "U", "P", "T", f"S{nb_vars[4]:02d}")
        variables = (str_vars[i] if nb_vars[i] > 0 else "" for i in range(5))
        return "".join(variables)

    def as_bytestring(self) -> bytes:
        header = "#std %1i %2i %2i %2i %10i %10i %20.13E %9i %6i %6i %s" % (
            self.wdsz,
            self.orders[0],
            self.orders[1],
            self.orders[2],
            self.nb_elems,
            self.nb_elems_file,
            self.time,
            self.istep,
            self.fid,
            self.nb_files,
            self.variables,
        )
        return header.ljust(132).encode("utf-8")


def read_header(path_or_file_obj: Union[PathLike, BinaryIO]) -> Header:
    """Make a :class:`pymech.neksuite.Header` instance from a file buffer
    opened in binary mode.

    """
    if isinstance(path_or_file_obj, (str, os.PathLike)):
        with Path(path_or_file_obj).open("rb") as fp:
            header = fp.read(132).split()
    elif isinstance(path_or_file_obj, io.BufferedReader):
        fp = path_or_file_obj
        header = fp.read(132).split()
    else:
        raise ValueError("Should be a path or opened file object in 'rb' mode.")

    logger.debug(b"Header: " + b" ".join(header))
    if len(header) < 12:
        raise IOError("Header of the file was too short.")

    # Relying on attrs converter to type-cast. Mypy will complain
    return Header(header[1], header[2:5], *header[5:12])  # type: ignore[arg-type]


# ==============================================================================
def readnek(fname, ref_values=None, dtype="float64", skip_vars=()):
    """A function for reading binary data from the nek5000 binary format

    Parameters
    ----------
    fname : str
        File name
    ref_values : list
        List with the reference values
    dtype : str or type
        Floating point data type. See also :class:`pymech.core.Elem`.
    skip_vars: tuple[str]
        Variables to skip. Valid values to skip are ``("x", "y", "z", "ux",
        "uy", "uz", "pressure", "temperature", "s01", "s02", ...)``.  It also
        accept some extra values ``("vx", "vy", "vz", "p", "t")``.  If empty
        (default), it reads all variables available in the file.

    """
    #
    if ref_values is None:
        ref_values = []
    try:
        infile = open(fname, "rb")
    except OSError as e:
        logger.critical(f"I/O error ({e.errno}): {e.strerror}")
        return -1
    #
    # ---------------------------------------------------------------------------
    # READ HEADER
    # ---------------------------------------------------------------------------
    #
    # Read header
    h = read_header(infile)
    #
    # Identify endian encoding
    etagb = infile.read(4)
    etagL = struct.unpack("<f", etagb)[0]
    etagL = int(etagL * 1e5) / 1e5
    etagB = struct.unpack(">f", etagb)[0]
    etagB = int(etagB * 1e5) / 1e5
    if etagL == 6.54321:
        logger.debug("Reading little-endian file\n")
        emode = "<"
    elif etagB == 6.54321:
        logger.debug("Reading big-endian file\n")
        emode = ">"
    else:
        logger.error("Could not interpret endianness")
        return -3
    #
    # Read element map for the file
    elmap = infile.read(4 * h.nb_elems_file)
    elmap = struct.unpack(emode + h.nb_elems_file * "i", elmap)
    #
    # ---------------------------------------------------------------------------
    # READ DATA
    # ---------------------------------------------------------------------------
    #
    # Initialize data structure
    if h.nb_dims == 3:
        ncellv = 8  # heaxahedron
    else:
        ncellv = 4  # quad
    data = ElemData(h.nb_dims, h.nb_elems, h.orders, h.nb_vars, ncellv, 0, dtype)
    data.time = h.time
    data.istep = h.istep
    data.wdsz = h.wdsz
    data.elmap = np.array(elmap, dtype=np.int32)
    if emode == "<":
        data.endian = "little"
    elif emode == ">":
        data.endian = "big"

    bytes_elem = h.nb_pts_elem * h.wdsz

    def read_file_into_data(data_var, index_var, ref_value):
        """Read binary file into an array attribute of ``data.elem``"""
        fi = infile.read(bytes_elem)
        fi = np.frombuffer(fi, dtype=emode + h.realtype, count=h.nb_pts_elem)

        # Replace elem array in-place with
        # array read from file after reshaping as
        # elem_shape = h.orders  # lx, ly, lz
        data_var[index_var, ...] = fi * ref_value

    def skip_elements(nb_elements=1):
        infile.seek(bytes_elem * nb_elements, os.SEEK_CUR)

    # Reference values defined in json file
    lRef, uRef, TRef, pRef, YRef = ref_values

    #
    # Read geometry
    #
    geometry_vars = "x", "y", "z"
    nb_vars = h.nb_vars[0]
    skip_condition = tuple(geometry_vars[idim] in skip_vars for idim in range(nb_vars))
    if nb_vars:
        if all(skip_condition):
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                ipt_old = (iel-1) * data.npel
                ipt_new = iel * data.npel
                el = data.pos[:, ipt_old:ipt_new]
                for idim in range(nb_vars):
                    if skip_condition[idim]:
                        skip_elements()
                    else:
                        read_file_into_data(el, idim, lRef)
    #
    # Read velocity
    #
    velocity_vars1 = "ux", "uy", "uz"
    velocity_vars2 = "vx", "vy", "vz"
    nb_vars = h.nb_vars[1]
    skip_condition1 = tuple(
        velocity_vars1[idim] in skip_vars for idim in range(nb_vars)
    )
    skip_condition2 = tuple(
        velocity_vars2[idim] in skip_vars for idim in range(nb_vars)
    )

    if nb_vars:
        if all(skip_condition1) or all(skip_condition2):
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                ipt_old = (iel-1) * data.npel
                ipt_new = iel * data.npel
                el = data.vel[:, ipt_old:ipt_new]
                for idim in range(nb_vars):
                    if skip_condition1[idim] or skip_condition2[idim]:
                        skip_elements()
                    else:
                        read_file_into_data(el, idim, uRef)
    #
    # Read pressure
    #
    nb_vars = h.nb_vars[2]
    skip_condition = any({"p", "pressure"}.intersection(skip_vars))
    if nb_vars:
        if skip_condition:
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                ipt_old = (iel-1) * data.npel
                ipt_new = iel * data.npel
                el = data.pres[:, ipt_old:ipt_new]
                for ivar in range(nb_vars):
                    read_file_into_data(el, ivar, pRef)
    #
    # Read temperature
    #
    nb_vars = h.nb_vars[3]
    skip_condition = any({"t", "temperature"}.intersection(skip_vars))
    if nb_vars:
        if skip_condition:
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                ipt_old = (iel-1) * data.npel
                ipt_new = iel * data.npel
                el = data.temp[:, ipt_old:ipt_new]
                for ivar in range(nb_vars):
                    read_file_into_data(el, ivar, TRef)
    #
    # Read scalar fields
    #
    nb_vars = h.nb_vars[4]
    scalar_vars = tuple(f"s{i:02d}" for i in range(1, nb_vars + 1))
    skip_condition = tuple(scalar_vars[ivar] in skip_vars for ivar in range(nb_vars))
    if nb_vars:
        if all(skip_condition):
            skip_elements(h.nb_elems * nb_vars)
        else:
            # NOTE: This is not a bug!
            # Unlike other variables, scalars are in the outer loop and elements
            # are in the inner loop
            for ivar in range(nb_vars):
                if skip_condition[ivar]:
                    skip_elements(h.nb_elems)
                else:
                    for iel in elmap:
                        ipt_old = (iel - 1) * data.npel
                        ipt_new = iel * data.npel
                        el = data.scal[:, ipt_old:ipt_new]
                        read_file_into_data(el, ivar, 1)
    #
    # Create cell vertices
    #
    lx1, ly1, lz1 = h.orders
    of = np.arange(data.nel) * data.npel
    ix_range = np.arange(lx1)
    iy_range = np.arange(ly1)
    iz_range = np.arange(lz1)
    ix, iy, iz, iel = np.ix_(ix_range[:-1], iy_range[:-1], iz_range[:-1], elmap)
    if h.nb_dims  == 3:
        data.cell_vert[:, 0] = (of + iz * ly1 * lx1 + iy * lx1 + ix).ravel()
        data.cell_vert[:, 1] = (of + iz * ly1 * lx1 + iy * lx1 + ix + 1).ravel()
        data.cell_vert[:, 2] = (of + iz * ly1 * lx1 + (iy + 1) * lx1 + ix + 1).ravel()
        data.cell_vert[:, 3] = (of + iz * ly1 * lx1 + (iy + 1) * lx1 + ix).ravel()
        data.cell_vert[:, 4] = (of + (iz + 1) * ly1 * lx1 + iy * lx1 + ix).ravel()
        data.cell_vert[:, 5] = (of + (iz + 1) * ly1 * lx1 + iy * lx1 + ix + 1).ravel()
        data.cell_vert[:, 6] = (of + (iz + 1) * ly1 * lx1 + (iy + 1) * lx1 + ix + 1).ravel()
        data.cell_vert[:, 7] = (of + (iz + 1) * ly1 * lx1 + (iy + 1) * lx1 + ix).ravel()
    else:
        data.cell_vert[:, 0] = (of + iy*lx1 + ix).ravel()
        data.cell_vert[:, 1] = (of + iy*lx1 + ix+1).ravel()
        data.cell_vert[:, 2] = (of + (iy+1)*lx1 + ix+1).ravel()
        data.cell_vert[:, 3] = (of + (iy+1)*lx1 + ix).ravel()
    #
    # Close file
    infile.close()
    #
    # Output
    return data


# ------------------------------- Read JSON file --------------------------------
print(f'\n>>>> Reading JSON file {fnamejson_in}', flush=True)
fjson = open(fnamejson_in)
data_json = json.load(fjson)
lRef = data_json["reference_length"]["value"]
uRef = data_json["reference_velocity"]["value"]
TRef = data_json["reference_temperature"]["value"]
pRef = data_json["reference_pressure"]["value"]
YRef = data_json["reference_mass_fractions"]["value"]
scalars_names = data_json["species"]["names"]
ref_values = [lRef, uRef, TRef, pRef, YRef]

# ------------------------------- Read Nek file ---------------------------------
print(f'\n>>>> Reading Nek5000 field file and preparing points for conversion: '
      f'{fnamenek_in}', flush=True)
t0 = time.time()
nekdata = readnek(fnamenek_in, ref_values)
t1 = time.time()
print(f'Time elapsed [sec]: {t1 - t0:.1f}', flush=True)

# ------------------------------- Write VTK file --------------------------------
if write_vtk_file:
    print(f'\n>>>> Writing VTK file: {fnamevtk_out}', flush=True)
    t0 = time.time()

    # points
    points = np.transpose(nekdata.pos)
    # cells
    if nekdata.ndim == 3:
        cell_type = "hexahedron"
    else:
        cell_type = "quad"
    cells = [(cell_type, nekdata.cell_vert)]
    # point data
    point_data = {}
    if nekdata.var[1] != 0:
        if nekdata.ndim == 3:
            point_data["x_velocity"] = nekdata.vel[0, :]
            point_data["y_velocity"] = nekdata.vel[1, :]
            point_data["z_velocity"] = nekdata.vel[2, :]
        else:
            point_data["x_velocity"] = nekdata.vel[0, :]
            point_data["y_velocity"] = nekdata.vel[1, :]
    if nekdata.var[2] != 0:
        point_data["pressure"] = nekdata.pres[0]
    if nekdata.var[3] != 0:
        point_data["temperature"] = nekdata.temp[0]
    if nekdata.var[4] != 0:
        for s in range(nekdata.var[4]):
            point_data[scalars_names[s]] = nekdata.scal[s, :]

    meshio.write_points_cells(
        fnamevtk_out,
        points,
        cells,
        point_data=point_data,
        file_format=vtk_file_format
    )

    t1 = time.time()
    print(f'Time elapsed [sec]: {t1 - t0:.1f}', flush=True)

# ------------------------------- Write HDF5 file -------------------------------
if write_hdf5_file:
    print(f'\n>>>> Write HDF5 file {fnamehdf5_out}', flush=True)
    t0 = time.time()

    h5f = h5py.File(fnamehdf5_out, 'w')
    if nekdata.var[0] != 0:
        if nekdata.ndim == 3:
            h5f.create_dataset('x_mesh', data=nekdata.pos[0, :])
            h5f.create_dataset('y_mesh', data=nekdata.pos[1, :])
            h5f.create_dataset('z_mesh', data=nekdata.pos[2, :])
        else:
            h5f.create_dataset('x_mesh', data=nekdata.pos[0, :])
            h5f.create_dataset('y_mesh', data=nekdata.pos[1, :])
    if nekdata.var[1] != 0:
        if nekdata.ndim == 3:
            h5f.create_dataset('x_velocity', data=nekdata.vel[0, :])
            h5f.create_dataset('y_velocity', data=nekdata.vel[1, :])
            h5f.create_dataset('z_velocity', data=nekdata.vel[2, :])
        else:
            h5f.create_dataset('x_velocity', data=nekdata.vel[0, :])
            h5f.create_dataset('y_velocity', data=nekdata.vel[1, :])
    if nekdata.var[2] != 0:
        h5f.create_dataset('pressure', data=nekdata.pres[0])
    if nekdata.var[3] != 0:
        h5f.create_dataset('temperature', data=nekdata.temp[0])
    if nekdata.var[4] != 0:
        for s in range(nekdata.var[4]):
            h5f.create_dataset(scalars_names[s], data=nekdata.scal[s, :])
    h5f.close()

    t1 = time.time()
    print(f'Time elapsed [sec]: {t1 - t0:.1f}', flush=True)

# -------------------------------------------------------------------------------
t11 = time.time()
print(f'\nTotal time elapsed [sec]: {t11 - t00:.1f}', flush=True)

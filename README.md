# nek2vtk

Convert Nek5000 binary field files to VTK or HDF5 formats with optional unit conversion.

This tool reads Nek5000's binary output files (.f***** format) and converts them to more popular formats for visualization and post-processing. It supports 2D and 3D simulations with both structured and unstructured mesh data, handles multiple field variables (velocity, pressure, temperature, and scalar fields), and can apply reference scaling for dimensional quantities when provided with a JSON configuration file. The converter preserves the spectral element mesh structure and interpolation points from the original Nek5000 data.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Basic Conversion

```bash
# Convert to VTK
python nek2vtk.py field0.f00001 --vtk

# Convert to HDF5
python nek2vtk.py field0.f00001 --hdf5

# Both formats
python nek2vtk.py field0.f00001 --vtk --hdf5
```

### With Unit Conversion

```bash
python nek5000_converter.py field0.f00001 --json-file reference.json --vtk
```

## Arguments

**Required:**
- `input_file`: Nek5000 field file (e.g., `test0.f00001`)

**Optional:**
- `--json-file`: JSON file with reference values for unit conversion
- `--vtk`: Write VTK output
- `--vtk-output`: Output filename (default: `output.vtk`)
- `--vtk-format`: Format [`vtk`, `vtk42`, `vtu`] (default: `vtk42`)
- `--hdf5`: Write HDF5 output
- `--hdf5-output`: Output filename (default: `output.h5`)
- `--skip-vars`: Variables to skip (e.g., `pressure temperature`)
- `--verbose`: Enable debug output

## JSON Reference File Example

```json
{
  "reference_length": {"value": 1.0},
  "reference_velocity": {"value": 1.0},
  "reference_temperature": {"value": 300.0},
  "reference_pressure": {"value": 101325.0},
  "reference_mass_fractions": {"value": 1.0},
  "species": {"names": ["O2", "N2", "H2O", "CO2"]}
}
```

Without JSON file, no unit conversion is applied (scaling factor = 1.0).

## Output

- **VTK**: Visualize with ParaView, VisIt
- **HDF5**: Process with h5py, MATLAB, or custom scripts

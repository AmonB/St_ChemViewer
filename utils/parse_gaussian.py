import re
import numpy as np
import streamlit as st
from typing import List, Dict
import pandas as pd


class GaussianParser:

    @staticmethod
    def parse_log_file(file_path: str):

        def parse_irc(content: str, results:dict):
            # 1. method and basis set
            method_pattern = r"SCF Done:\s+E\((\w+)\)\s+=\s+(-?\d+\.\d+)"
            method_match = re.search(method_pattern, content)
            if method_match:
                results['method'] =  method_match.group(1).strip()

            basis_match = re.search(r"Standard basis:\s*(\w+)\s*\(.*?\)", content)
            if basis_match:
                results['basis_set'] = basis_match.group(1).strip()

            stoichiometry_match = re.search(r'Stoichiometry\s+(\S+)', content)
            if stoichiometry_match:
                results['stoichiometry'] = stoichiometry_match.group(1)

            natoms_match = re.search(r"NAtoms=\s*(\d+)", content)
            if natoms_match:
                results['num_atoms'] = natoms_match.group(1)

            # 2. charge and multiplicity
            charge_mult_match = re.search(r'Charge =\s*(\S+)\s*Multiplicity =\s*(\S+)', content)
            if charge_mult_match:
                results['charges'] = int(charge_mult_match.group(1))
                results['multiplicity'] = int(charge_mult_match.group(2))

            # 3. tmp for sort frames
            temp_geom = []
            tmp_energies = []

            # is read from chk
            read_from_chk_marker = r"Redundant internal coordinates(.*?)Recover connectivity data from disk\."
            read_from_chk_match = re.search(read_from_chk_marker, content,re.DOTALL)
            energy_match = re.search(r'Energy From Chk =\s*(\S+)', content)
            if energy_match:
                tmp_energies.append(float(energy_match.group(1)))

            if read_from_chk_match:
                frame_atoms = []
                lines = read_from_chk_match.group(1).strip().split('\n')[1:]

                for line in lines:
                    parts = line.split(',')
                    try:
                        x, y, z = map(float, parts[2:5])
                        element = parts[0]
                        frame_atoms.append({
                            'element': element,
                            'x': x,
                            'y': y,
                            'z': z,
                            'atomic_number': GaussianParser._get_atmoic_number(element),
                        })
                    except:
                        continue

                if frame_atoms:
                    temp_geom.append(frame_atoms)


            input_orient_pattern = r'Input orientation:(.*?)(?=Rotational|I=|\\\\)'
            orientation_matches = re.findall(input_orient_pattern, content, re.DOTALL)

            for match_idx, match in enumerate(orientation_matches):
                frame_atoms = []
                lines = match.strip().split('\n')

                for line in lines:
                    if line.strip() and '---' not in line:
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                atom_num = int(parts[1])
                                x, y, z = map(float, parts[3:6])
                                element = GaussianParser._get_element_symbol(atom_num)
                                frame_atoms.append({
                                    'element': element,
                                    'x': x,
                                    'y': y,
                                    'z': z,
                                    'atomic_number': atom_num
                                })
                            except:
                                continue

                if frame_atoms:
                    temp_geom.append(frame_atoms)

            # 4. energies
            scf_pattern = r'SCF Done:\s*E\(.+?\)\s*=\s*(-?\d+\.\d+)'
            scf_energies = re.findall(scf_pattern, content)

            energies = scf_energies
            for i, energy in enumerate(energies):
                tmp_energies.append(energy)

            # 5. sort path
            path_point_pattern = r'Point Number:\s*(\d+)\s*Path Number:\s*(\d+)'
            path_point_matches = re.findall(path_point_pattern, content, re.DOTALL)
            path_point_numbers = len(path_point_matches)

            path_two_point_pattern = r'Point Number:\s*(\d+)\s*Path Number:\s*(2)'
            path_two_point_matches = re.findall(path_two_point_pattern, content, re.DOTALL)
            path_two_point_numbers = len(path_two_point_matches)
            path_one_point_numbers = path_point_numbers - path_two_point_numbers


            if path_two_point_numbers:
                for i in range (path_two_point_numbers):
                    results['trajectory'].append(temp_geom[path_point_numbers-1 - i])
                    results['energies'].append({
                        'step': i + 1,
                        'energy': float(tmp_energies[path_point_numbers-1 - i]),
                        'energy_kcal': float(tmp_energies[path_point_numbers-1 -i]) * 627.5095  # Hartree转kcal/mol
                    })


            for i in range (path_one_point_numbers):
                results['trajectory'].append(temp_geom[i])
                results['energies'].append({
                    'step': path_two_point_numbers + i+1,
                    'energy': float(tmp_energies[i]),
                    'energy_kcal': float(tmp_energies[i]) * 627.5095  # Hartree转kcal/mol
                })

            mask_frame_atoms = [[{'element': 'H',
                            'x': 0.0,
                            'y': 0.0,
                            'z': 0.0,
                            'atomic_number': 1,}]]
            results['trajectory'] = mask_frame_atoms + results['trajectory']


            # 6. RxCoord
            rxcoord_marker = r"Summary of reaction path following(.*?) Total number of points:"
            rxcoord_matches = re.search(rxcoord_marker, content, re.DOTALL)
            if rxcoord_matches:
                rx_coords  = []
                rxcoord_lines = rxcoord_matches.group(1).strip().split('\n')[2:-1]
                # st.info(rxcoord_lines)
                for line in rxcoord_lines:
                    parts = line.split(' ')
                    rx_coords.append(float(parts[-1]))

                results['rx_coords'] = rx_coords
                # st.info(results['rx_coords'])

            return results


        def parse_opt(content:str, results:dict):
            # 1. method and basis set
            method_pattern = r"SCF Done:\s+E\((\w+)\)\s+=\s+(-?\d+\.\d+)"
            method_match = re.search(method_pattern, content)
            if method_match:
                results['method'] =  method_match.group(1).strip()

            basis_match = re.search(r'Standard basis:\s*(.*?)', content, re.DOTALL)
            if basis_match:
                results['basis_set'] = basis_match.group(1).strip()

            stoichiometry_match = re.search(r'Stoichiometry\s+(\S+)', content)
            if stoichiometry_match:
                results['stoichiometry'] = stoichiometry_match.group(1)

            natoms_match = re.search(r"NAtoms=\s*(\d+)", content)
            if natoms_match:
                results['num_atoms'] = natoms_match.group(1)

            # 2. charge and multiplicity
            charge_mult_match = re.search(r'Charge =\s*(\S+)\s*Multiplicity =\s*(\S+)', content)
            if charge_mult_match:
                results['charges'] = int(charge_mult_match.group(1))
                results['multiplicity'] = int(charge_mult_match.group(2))


            # 3. xyz data
            orientation_pattern = r'Standard orientation:(.*?)(?=Rotational|I=|\\\\)'
            input_orient_pattern = r'Input orientation:(.*?)(?=Rotational|I=|\\\\)'
            input_orient_matches = re.findall(input_orient_pattern, content, re.DOTALL)
            orientation_matches = re.findall(orientation_pattern, content, re.DOTALL)
            if orientation_matches:
                pass
            else:
                orientation_matches = input_orient_matches

            for match_idx, match in enumerate(orientation_matches):
                frame_atoms = []
                lines = match.strip().split('\n')

                for line in lines:
                    if line.strip() and '---' not in line:
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                atom_num = int(parts[1])
                                x, y, z = map(float, parts[3:6])
                                element = GaussianParser._get_element_symbol(atom_num)
                                frame_atoms.append({
                                    'element': element,
                                    'x': x,
                                    'y': y,
                                    'z': z,
                                    'atomic_number': atom_num
                                })
                            except:
                                continue

                if frame_atoms:
                    results['trajectory'].append(frame_atoms)

            # 4. energies
            scf_pattern = r'SCF Done:\s*E\(.+?\)\s*=\s*(-?\d+\.\d+)'
            scf_energies = re.findall(scf_pattern, content)

            energies = scf_energies
            for i, energy in enumerate(energies):
                results['energies'].append({
                    'step': i + 1,
                    'energy': float(energy),
                    'energy_kcal': float(energy) * 627.5095  # Hartree转kcal/mol
                })

            # 5. RMS Gradient Norm
            gradient_pattern = r' Cartesian Forces:\s+Max\s+([\d\.]+)\s+RMS\s+([\d\.]+)'
            gradient_matches = re.findall(gradient_pattern, content)
            for i, match in enumerate(gradient_matches):
                results['gradients'].append({
                    'step': i + 1,
                    'gradients': float(match[1]),
                })

            # 5. Max Force
            force_pattern = r'Maximum Force\s+(-?\d+\.\d+E?[-+]?\d*)\s+(\S+)'
            force_matches = re.findall(force_pattern, content)

            for i, match in enumerate(force_matches):
                results['max_forces'].append({
                    'step': i + 1,
                    'forces': float(match[0]),
                    'converged': "YES" if float(match[0]) < float(match[1]) else "NO"
                })

            # 6. RMS force
            rms_pattern = r'RMS\s+Force\s+(-?\d+\.\d+E?[-+]?\d*)\s+(\S+)'
            rms_matches = re.findall(rms_pattern, content)

            for i, match in enumerate(rms_matches):
                results['rms_force'].append({
                    'step': i + 1,
                    'rms_force': float(match[0]),
                    'converged': "YES" if float(match[0]) < float(match[1]) else "NO"
                })

            # 7. Displacement
            displacement_pattern = r'Maximum Displacement\s+(-?\d+\.\d+E?[-+]?\d*)\s+(\S+)'
            displacement_matches = re.findall(displacement_pattern, content)

            for i, match in enumerate(displacement_matches):
                results['displacements'].append({
                    'step': i + 1,
                    'displacements': float(match[0]),
                    'converged': "YES" if float(match[0]) < float(match[1]) else "NO"
                })

            # 8. RMS Displacement
            rms_displacement_pattern = r'RMS\s+Displacement\s+(-?\d+\.\d+E?[-+]?\d*)\s+(\S+)'
            rms_displacement_matches = re.findall(rms_displacement_pattern, content)

            for i, match in enumerate(rms_displacement_matches):
                results['rms_displacements'].append({
                    'step': i + 1,
                    'rms_displacements': float(match[0]),
                    'converged': "YES" if float(match[0]) < float(match[1]) else "NO"
                })

            # 9. Thermochemisty
            temperature_pressure_pattern = r'Temperature\s+([\d.]+).*?Pressure\s+([\d.]+)'
            temperature_pressure_matches = re.search(temperature_pressure_pattern, content)
            if temperature_pressure_matches:
                results['temperature'] = float(temperature_pressure_matches.group(1))
                results['pressure'] = float(temperature_pressure_matches.group(2))

            zpe_pattern = r"Zero-point correction=\s*(-?\d+\.?\d*)"
            zpe_match = re.search(zpe_pattern, content)
            if zpe_match:
                results['zpe'] = float(zpe_match.group(1))

            therm_correct_energy_pattern = r"Thermal correction to Energy=\s*(-?\d+\.?\d*)"
            therm_correct_energy_match = re.search(therm_correct_energy_pattern, content)
            if therm_correct_energy_match:
                results['tce'] = float(therm_correct_energy_match.group(1))

            therm_correct_enthalpy_pattern = r'Thermal correction to Enthalpy=\s*(-?\d+\.?\d*)'
            therm_correct_enthalpy_match = re.search(therm_correct_enthalpy_pattern, content)
            if therm_correct_enthalpy_match:
                results['tch'] = float(therm_correct_enthalpy_match.group(1))

            therm_correct_gibbs_pattern = r'Thermal correction to Gibbs Free Energy=\s*(-?\d+\.?\d*)'
            therm_correct_gibbs_match = re.search(therm_correct_gibbs_pattern, content)
            if therm_correct_gibbs_match:
                results['tcg'] = float(therm_correct_gibbs_match.group(1))

            sum_zpe_pattern = r'Sum of electronic and zero-point Energies=\s*(-?\d+\.?\d*)'
            sum_zpe_match = re.search(sum_zpe_pattern, content)
            if sum_zpe_match:
                results['sum_zpe'] = float(sum_zpe_match.group(1))

            sum_tce_pattern = r'Sum of electronic and thermal Energies=\s*(-?\d+\.?\d*)'
            sum_tce_match = re.search(sum_tce_pattern, content)
            if sum_tce_match:
                results['sum_tce'] = float(sum_tce_match.group(1))

            sum_tch_pattern = r'Sum of electronic and thermal Enthalpies=\s*(-?\d+\.?\d*)'
            sum_tch_match = re.search(sum_tch_pattern, content)
            if sum_tch_match:
                results['sum_tch'] = float(sum_tch_match.group(1))

            sum_tcg_pattern = r'Sum of electronic and thermal Free Energies=\s*(-?\d+\.?\d*)'
            sum_tcg_match = re.search(sum_tcg_pattern, content)
            if sum_tcg_match:
                results['sum_tcg'] = float(sum_tcg_match.group(1))


            # 10. Frequencies and vectors
            freq_pattern = r'Frequencies --\s*([-\d\.\s]+)'
            freq_blocks = re.findall(freq_pattern, content)
            freqs_all = []
            for block in freq_blocks:
                try:
                    nums = [float(x) for x in block.split() if x.strip()]
                    freqs_all.extend(nums)
                except:
                    continue

            results['frequencies'] = freqs_all
            results['modes'] = []

            float_pat = r'[-+]?\d*\.\d+|[-+]?\d+'
            content_lines = content.splitlines()
            freq_line_nums = []
            for i, line in enumerate(content_lines):
                if 'Frequencies --' in line:
                    freq_line_nums.append(i)

            for ln in freq_line_nums:
                line = content_lines[ln]
                try:
                    block_vals = [float(x) for x in re.findall(float_pat, line)]
                    freqs_here = block_vals
                except:
                    freqs_here = []

                n_modes = len(freqs_here)
                if n_modes == 0:
                    continue

                search_window = 200
                table_lines = []
                for j in range(ln + 1, min(len(content_lines), ln + 1 + search_window)):
                    l = content_lines[j].strip()
                    if not l:
                        continue
                    if re.match(r'^\d+\s+\w+', l):
                        table_lines.append(content_lines[j])
                    else:
                        if 'Normal coordinates' in l or l.startswith('Atom') or l.startswith('--'):
                            table_lines.append(content_lines[j])
                        if len(table_lines) > 0 and not re.match(r'^\d+\s+\w+', l):
                            pass

                if not table_lines:
                    continue

                atom_elements = []
                per_mode_vectors = [[] for _ in range(n_modes)]

                for tl in table_lines:
                    parts = tl.strip().split()
                    if len(parts) < 4:
                        continue
                    if not re.match(r'^\d+$', parts[0]):
                        continue
                    elem = parts[1]
                    nums = re.findall(float_pat, ' '.join(parts[2:]))
                    if len(nums) < 3:
                        continue
                    atom_elements.append(elem)
                    needed = 3 * n_modes
                    if len(nums) < needed:
                        nums = nums + ['0.0'] * (needed - len(nums))
                    for m in range(n_modes):
                        dx = float(nums[3 * m])
                        dy = float(nums[3 * m + 1])
                        dz = float(nums[3 * m + 2])
                        per_mode_vectors[m].append((dx, dy, dz))

                for m_idx, freq_val in enumerate(freqs_here):
                    mode = {
                        'freq': float(freq_val),
                        'vectors': per_mode_vectors[m_idx],
                        'atoms': atom_elements
                    }
                    results['modes'].append(mode)

            mask_frame_atoms = [[{'element': 'H',
                                  'x': 0.0,
                                  'y': 0.0,
                                  'z': 0.0,
                                  'atomic_number': 1, }]]
            results['trajectory'] = mask_frame_atoms + results['trajectory']
            return results


        # parse start
        stop_read_irc_marker = "Total number of gradient calculations:"
        content = []
        is_irc_type = False
        try:

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    content.append(line)
                    if 'IRC-IRC-IRC-' in line:
                        is_irc_type = True
                        st.session_state.parsed_irc = False
                    if stop_read_irc_marker in line:
                        break
                content = ''.join(content)


            results = {
                'trajectory': [],
                'energies': [],
                'gradients': [],
                'max_forces': [],
                'rms_force': [],
                'displacements': [],
                'rms_displacements': [],
                'charges': [],
                'multiplicity': [],
                'method': '',
                'basis_set': '',
                'stoichiometry': '',
                'converge_thread': [],
                'modes': [],
                'temperature': [],
                'pressure': [],
                'rx_coords': [],
                'zpe': [],
                'tce': [],
                'tch': [],
                'tcg': [],
                'sum_zpe': [],
                'sum_tce': [],
                'sum_tch': [],
                'sum_tcg': [],

            }

            if not is_irc_type:
                results = parse_opt(content, results)
                st.session_state.parsed_irc = False
            else:
                results = parse_irc(content, results)
                st.session_state.parsed_irc = True

            return results
        except Exception as e:
            st.error(f"Parse Error: {e}")
            return None

    @staticmethod
    def _get_element_symbol(atomic_number: int) -> str:
        elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        ]
        if 1 <= atomic_number <= len(elements):
            return elements[atomic_number - 1]
        return 'X'


    @staticmethod
    def _get_atmoic_number(element_symbol: str) -> int:
        elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        ]
        element_symbol = element_symbol.strip()
        symbol = element_symbol[0].upper()
        if len(element_symbol) > 1:
            symbol += element_symbol[1].lower()
        return elements.index(symbol) + 1


    @staticmethod
    def create_xyz_string(atoms: List[Dict]) -> str:
        xyz = f"{len(atoms)}\nGenerated by Gaussian Visualizer\n"
        for atom in atoms:
            xyz += f"{atom['element']} {atom['x']:.8f} {atom['y']:.8f} {atom['z']:.8f}\n"
        xyz += f"\n"
        return xyz

    @staticmethod
    def parse_input_file(file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            results = {
                'trajectory': [],


            }

            pattern = r'([A-Z][a-z]?)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)'
            pattern2 =  r'([A-Z][a-z]?),([-]?\d+\.\d+)\,([-]?\d+\.\d+)\,([-]?\d+\.\d+)'
            matches = re.findall(pattern, content)
            matches_pattern_2 = re.findall(pattern2, content)
            if matches:
                pass
            else:
                matches = matches_pattern_2
            atoms = []
            for match in matches:
                element, x, y, z = match[0], float(match[1]), float(match[2]), float(match[3])
                atoms.append({
                    'element': element,
                    'x': x,
                    'y': y,
                    'z': z,
                })
            if atoms:
                results['trajectory'].append(atoms)
            mask_frame_atoms = [[{'element': 'H',
                            'x': 0.0,
                            'y': 0.0,
                            'z': 0.0,
                            'atomic_number': 1,}]]
            results['trajectory'] = mask_frame_atoms + results['trajectory']
            return results

        except Exception as e:
            st.error(f"解析文件时出错: {e}")
            return None


    @staticmethod
    def create_input_info_table(atoms: List[Dict]):
        """creat info table"""
        if not atoms:
            return

        try:
            # get element list
            atom_counts = {}
            for atom in atoms:
                element = atom['element']
                atom_counts[element] = atom_counts.get(element, 0) + 1

            # generate formula
            formula = ''.join([f"{element}{count if count > 1 else ''}"
                               for element, count in atom_counts.items()])

            info_data = pd.DataFrame({
                'Attribution': [' Stoichiometry', 'Number of atoms'],
                'Values': [str(formula), str(len(atoms))]
            })


            st.dataframe(info_data, width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Creat info table error: {e}")


    @staticmethod
    def create_molecule_info_table(atoms: List[Dict], stoichiometry: str, num_atoms: str,
                                   method: str, charge: float, multiplicity: float
                                   ):
        """creat info table"""
        if not atoms:
            return

        try:
            if stoichiometry is None:
            # get element list
                atom_counts = {}
                for atom in atoms:
                    element = atom['element']
                    atom_counts[element] = atom_counts.get(element, 0) + 1

                # generate formula
                formula = ''.join([f"{element}{count if count > 1 else ''}"
                                   for element, count in atom_counts.items()])
                formula = str(formula)

            else:
                formula = str(stoichiometry)

            if num_atoms is None:
                natoms = str(len(atoms))
            else:
                natoms = num_atoms


            if method is None:
                method = np.nan
            else:
                method = str(method[0])
            # if basis_set is None:
            #     basis_set = np.nan
            # else:
            #     basis_set = str(basis_set[0])
            #     if basis_set == "":
            #         basis_set = "Unknown"

            if charge is None:
                charge = np.nan
            else:
                charge = str(charge[0])

            if multiplicity is None:
                multiplicity = np.nan
            else:
                multiplicity = str(multiplicity)

            info_data = pd.DataFrame({
                'Attribution': ['Calculation Method', 'Charge', 'Multiplicity',
                                'Stoichiometry', 'Number of atoms', ],
                'Values': [method, charge, multiplicity, formula, natoms]
            })

            info_data = info_data.dropna()
            st.dataframe(info_data, width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Creat info table error: {e}")


    @staticmethod
    def creat_thermo_info_table(temperature: str, pressure: str, electronic_energy: str,
                                zpe: str, tce: str, tch: str, tcg: str,
                                sum_zpe: str, sum_tce: str, sum_tch: str, sum_tcg: str):

        try:
            thermo_data = pd.DataFrame({
                'Attribution': [
                    'Temperature', 'pressure', 'Electronic Energy (EE)', 'Zero-point Energy Correction',
                    'Thermal Correction to Energy', 'Thermal Correction to Enthalpy',
                    'Thermal Correction to Free Energy', 'EE + Zero-point Energy',
                    'EE + Thermal Energy Correction', 'EE + Thermal Enthalpy Correction',
                    'EE + Thermal Free Energy Correction'],
                'Values': [temperature, pressure, electronic_energy, zpe, tce, tch, tcg,
                           sum_zpe, sum_tce, sum_tch, sum_tcg],
                'Unit': ['Kelvin', 'atm', 'Hartree', 'Hartree', 'Hartree', 'Hartree', 'Hartree',
                         'Hartree', 'Hartree', 'Hartree', 'Hartree']
            })

            st.dataframe(thermo_data, width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Export thermochemistry data error: {e}")


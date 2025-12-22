import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.opt_monitor import OptimizationMonitor
from utils.mol_viewer import MoleculeVisualizer
from utils.parse_gaussian import GaussianParser

# py3Dmol will be used via embedded 3Dmol.js in HTML
# keep existing HAS_3D check
try:
    import py3Dmol
    HAS_3D = True
except ImportError as e:
    st.warning(f"3D viewer ImportError: {e}")
    HAS_3D = False


# page setting
st.set_page_config(
    page_title="Gaussian visualization tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS style
st.markdown("""
<style>

    /* mol-viewer-container */
    .mol-viewer-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        width: 100%;

}

    /* vib-animation-container */
    .vib-animation-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        width: 100%;
        position: relative;
}

    /* unified-3d-wrapper */
    .unified-3d-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: white;
        overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# initial session state
def init_session_state():
    defaults = {
        'view_file': False,
        'current_frame': 1,
        'total_frames': 0,
        'trajectory_data': None,
        'energy_data': None,
        'gradient_data': None,
        'force_data': None,
        'rms_force_data': None,
        'displacement_data': None,
        'rms_displacement_data': None,
        'irc_rx_coord_data': None,
        'working_dir': os.getcwd(),
        'selected_file': None,
        'parsed': False,
        'view_style': 'ballstick',
        'rotation_enabled': False,
        'rotation_axis': 'y',
        'view_width': 800,
        'view_height': 500,
        'is_playing': False,
        'playback_speed': 10,
        'parsed_irc': False,
        'parsed_gjf': False,
        'method': None,
        'basis_set': None,
        'charges': None,
        'multiplicity': None,
        'stoichiometry': None,
        'num_atoms': None,
        'temperature': None,
        'pressure': None,
        'zpe': None,
        'tce': None,
        'tch': None,
        'tcg': None,
        'sum_zpe': None,
        'sum_tce': None,
        'sum_tch': None,
        'sum_tcg': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()



class PageSetUp:
    """Sidebar components"""
    @staticmethod
    def render_sidebar():

        # Select Directory
        with st.sidebar.expander("📂 Select Directory", expanded=True):
            current_dir = st.text_input(
                "Working directory",
                value=st.session_state.working_dir,
                help="Input Gaussian files directory"
            )

            if os.path.exists(current_dir):
                st.session_state.working_dir = current_dir
                # list files
                try:
                    files = os.listdir(current_dir)
                    gaussian_files = [f for f in files if f.lower().endswith(('.log', '.out', '.gjf', '.com'))]

                    if gaussian_files:
                        selected_file = st.selectbox(
                            "Choose a file",
                            gaussian_files,
                            index=0 if st.session_state.selected_file not in gaussian_files else
                            gaussian_files.index(st.session_state.selected_file),
                            help="Supported: .log, .out, .gjf, .com"
                        )

                        st.session_state.selected_file = selected_file

                        # file info
                        file_path = os.path.join(current_dir, selected_file)
                        file_size = os.path.getsize(file_path) / 1024  # KB

                        st.caption(f"📄 **{selected_file}**")
                        st.caption(f"📏 Size: {file_size:.1f} KB")
                        st.caption(f"📁 Path: {os.path.dirname(file_path)}")


                        if st.button("🔍 Decipher ", type="primary", width='stretch'):
                            if st.session_state.selected_file.endswith(('.gjf', '.com')):
                                with st.spinner("Parsing file..."):
                                    results = GaussianParser.parse_input_file(file_path)
                                    if len(results['trajectory']) >1:
                                        st.session_state.trajectory_data = results['trajectory']
                                        st.session_state.force_data = None
                                        st.session_state.sum_tcg = None
                                        st.session_state.energy_data = None
                                        st.session_state.total_frames = 1
                                        st.session_state.parsed = True
                                        st.session_state.parsed_gjf = True
                                        st.rerun()
                                    else:
                                        st.session_state.parsed = False
                                        st.error("❌ Geometry Not Found")

                            else:
                                with st.spinner("Parsing file..."):
                                    results = GaussianParser.parse_log_file(file_path)
                                    if len(results['trajectory']) >1:
                                        st.session_state.trajectory_data = results['trajectory']
                                        st.session_state.total_frames = len(results['energies'])
                                        st.session_state.energy_data = results['energies']
                                        st.session_state.gradient_data = results['gradients']
                                        st.session_state.force_data = results['max_forces']
                                        st.session_state.rms_force_data = results['rms_force']
                                        st.session_state.displacement_data = results['displacements']
                                        st.session_state.rms_displacement_data = results['rms_displacements']
                                        st.session_state.frequencies = results.get('frequencies', [])
                                        st.session_state.modes = results.get('modes', [])
                                        # st.session_state.current_frame =  len(results['energies']) -1
                                        st.session_state.current_frame = len(results['energies'])
                                        st.session_state.irc_rx_coord_data = results['rx_coords']
                                        st.session_state.parsed = True
                                        st.session_state.parsed_gjf = False


                                        st.session_state.method = results['method'],
                                        st.session_state.basis_set = results['basis_set'],
                                        st.session_state.charges = results['charges'],
                                        st.session_state.multiplicity = results['multiplicity']
                                        st.session_state.stoichiometry = results['stoichiometry']
                                        st.session_state.num_atoms = results['num_atoms']

                                        # thermochemistry data
                                        st.session_state.temperature = results['temperature']
                                        st.session_state.pressure = results['pressure']
                                        st.session_state.zpe = results['zpe']
                                        st.session_state.tce = results['tce']
                                        st.session_state.tch = results['tch']
                                        st.session_state.tcg = results['tcg']
                                        st.session_state.sum_zpe = results['sum_zpe']
                                        st.session_state.sum_tce = results['sum_tce']
                                        st.session_state.sum_tch = results['sum_tch']
                                        st.session_state.sum_tcg = results['sum_tcg']

                                        st.rerun()

                                    else:
                                        st.session_state.parsed = False
                                        st.error("❌ Oh crap! T^T")
                    else:
                        st.info("📭 No Gaussian files")

                except Exception as e:
                    st.error(f"❌ Error reading Directory: {e}")
            else:
                st.error("❌ Directory doesn't exist")


        # visualization settings
        with st.sidebar.expander("⚙️ Visualization", expanded=False):
            if HAS_3D:
                st.markdown("##### Viewer size")

                col1, col2 = st.columns(2)
                with col1:
                    view_width = st.slider(
                        "Width (piex)",
                        min_value=400,
                        max_value=1600,
                        value=st.session_state.view_width,
                        step=50,
                        help="Adjust view window width"
                    )
                with col2:
                    view_height = st.slider(
                        "Height (piex)",
                        min_value=300,
                        max_value=1200,
                        value=st.session_state.view_height,
                        step=50,
                        help="Adjust view window height"
                    )

                # save to session_state
                st.session_state.view_width = view_width
                st.session_state.view_height = view_height

                # preset size options
                st.markdown("##### Preset Viewer size")
                preset_col1, preset_col2, preset_col3 = st.columns(3)

                with preset_col1:
                    if st.button("S", width='stretch', help="600×400像素"):
                        st.session_state.view_width = 600
                        st.session_state.view_height = 400
                        st.rerun()

                with preset_col2:
                    if st.button("M", width='stretch', help="800×500像素"):
                        st.session_state.view_width = 800
                        st.session_state.view_height = 500
                        st.rerun()

                with preset_col3:
                    if st.button("L", width='stretch', help="1000×600像素"):
                        st.session_state.view_width = 1000
                        st.session_state.view_height = 600
                        st.rerun()

                # molecule style
                view_style = st.selectbox(
                    "Molecule Style",
                    options=['stick', 'sphere', 'ballstick', 'line'],
                    index=['stick', 'sphere', 'ballstick', 'line'].index(st.session_state.view_style),
                    help="Select to style to show mol."
                )
                st.session_state.view_style = view_style

            else:
                st.warning("3D viewer not installed")


            # st.markdown("#### Rotation by")
            rotation_enabled = st.checkbox(
                "Rotation",
                value=st.session_state.rotation_enabled,
            )
            if rotation_enabled:
                preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
                with preset_col1:
                    x_rotation = st.button("X", width='stretch', key="x_rotation", help='X-axis')
                with preset_col2:
                    y_rotation = st.button("Y", width='stretch', key="y_rotation", help='Y-axis')
                with preset_col3:
                    z_rotation = st.button("Z", width='stretch', key="z_rotation", help='Z-axis')
                with preset_col4:
                    stop_roatation = st.button("Stop", key="stop_rotation", help='Stop')
                if x_rotation:
                    st.session_state['rotation_axis'] = 'x'
                    st.session_state['rotation_enabled'] = True
                if y_rotation:
                    st.session_state['rotation_axis'] = 'y'
                    st.session_state['rotation_enabled'] = True
                if z_rotation:
                    st.session_state['rotation_axis'] = 'z'
                    st.session_state['rotation_enabled'] = True
                if stop_roatation:
                    st.session_state['rotation_enabled'] = False
            else:
                st.session_state.rotation_enabled = False



        with st.sidebar.expander("🎵 Vibration", expanded=False):
            if not st.session_state.get('parsed', False):
                st.info("Please parse Gaussian output file to get vibrations")
            else:
                freqs = st.session_state.get('frequencies', [])
                modes = st.session_state.get('modes', [])

                if not freqs:
                    st.info("Frequency data are not available.")
                else:

                    show_neg_only = st.checkbox("Imaginary Frequency only", value=True)
                    if show_neg_only:
                        list_freqs = [m['freq'] for m in modes if m['freq'] < 0] if modes else [f for f in freqs if
                                                                                                f < 0]
                    else:
                        list_freqs = [m['freq'] for m in modes] if modes else freqs

                    if not list_freqs:
                        st.info("No Imaginary Frequency")
                    else:
                        st.markdown("##### Frequencies (cm⁻¹)")
                        st.write(np.array(list_freqs))

                        # Select Freq to show
                        sel_idx = st.selectbox("Select Freq to show",
                                               options=list(range(len(list_freqs))),
                                               format_func=lambda i: f"{i + 1} : {list_freqs[i]:.2f} cm⁻¹")

                        # play control
                        apt_col, period_col = st.columns(2)
                        with apt_col:
                            amplitude = st.slider("Aptitude", min_value=0.01, max_value=2.0, value=0.3, step=0.01)
                        with period_col:
                            period = st.slider("Period (s)", min_value=0.2, max_value=5.0, value=2.0, step=0.1)

                        # vector arrow settings
                        show_arrows = st.checkbox("Show vectors", value=False)
                        if show_arrows:
                            arrow_length = st.slider("Arrow length", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
                            arrow_radius = st.slider("Arrow size", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
                        else:
                            arrow_length = 0.05
                            arrow_radius = 0.05

                        fps = st.slider("FPS", min_value=15, max_value=60, value=24, step=1,
                                        help="Lower FPS for higher performance")

                        # play and pause
                        col_play, col_pause = st.columns(2)
                        with col_play:
                            play_btn = st.button("▶️ play", width='stretch', key="play_vib")


                        with col_pause:
                            pause_btn = st.button("⏸️ stop", width='stretch', key="pause_vib")

                        # map selected index
                        mode_to_play = None
                        if modes:
                            target_freq = list_freqs[sel_idx]
                            for m in modes:
                                if abs(m['freq'] - target_freq) < 1e-3 and m.get('vectors'):
                                    mode_to_play = m
                                    break

                        if play_btn:
                            if mode_to_play is None:
                                st.error(
                                    "Vectors not found. Please check this part -- Normal coordinates")
                                st.error(print(modes))
                            else:
                                st.session_state['vib_mode_to_play'] = mode_to_play
                                st.session_state['vib_amplitude'] = amplitude
                                st.session_state['vib_period'] = period
                                st.session_state['vib_show_arrows'] = show_arrows
                                st.session_state['vib_arrow_length'] = arrow_length
                                st.session_state['vib_arrow_radius'] = arrow_radius
                                st.session_state['vib_fps'] = fps
                                st.session_state['vib_playing'] = True
                                st.session_state['vib_arrow_length'] = arrow_length
                                st.session_state['vib_arrow_radius'] = arrow_radius
                                MoleculeVisualizer.display_trajectory_player_with_3d(
                                    trajectory_data=st.session_state.trajectory_data,
                                    current_frame=st.session_state.current_frame,
                                    total_frames=st.session_state.total_frames,
                                    style=st.session_state.view_style,
                                    width=st.session_state.view_width,
                                    height=st.session_state.view_height,
                                    rotation_enabled=st.session_state.rotation_enabled,
                                    rotation_axis=st.session_state.get('rotation_axis', 'y'),
                                    playback_speed=st.session_state.get('playback_speed', 10),
                                )
                                st.rerun()

                        if pause_btn:
                            st.session_state['vib_playing'] = False
                            MoleculeVisualizer.display_trajectory_player_with_3d(
                                trajectory_data=st.session_state.trajectory_data,
                                current_frame=st.session_state.current_frame,
                                total_frames=st.session_state.total_frames,
                                style=st.session_state.view_style,
                                width=st.session_state.view_width,
                                height=st.session_state.view_height,
                                rotation_enabled=st.session_state.rotation_enabled,
                                rotation_axis=st.session_state.get('rotation_axis', 'y'),
                                playback_speed=st.session_state.get('playback_speed', 10),
                            )
                            st.rerun()

        # Export data
        with st.sidebar.expander("💾 Export Data", expanded=False):
            # export current data
            if st.button("📥 Export current XYZ", width='stretch'):
                if st.session_state.trajectory_data:
                    try:
                        current_atoms = st.session_state.trajectory_data[st.session_state.current_frame]
                        xyz_str = GaussianParser.create_xyz_string(current_atoms)
                        st.sidebar.download_button(
                            label="download XYZ file",
                            data=xyz_str,
                            file_name=f"frame_{st.session_state.current_frame}.xyz",
                            mime="chemical/x-xyz",
                            width='stretch'
                        )
                    except Exception as e:
                        st.error(f"Export XYZ failure: {e}")

            # export trajectory
            if st.button("📊 Export trajectory", width='stretch'):
                if st.session_state.trajectory_data:
                    try:
                        all_xyz = ""
                        # since we got a mask frame,here start from index 1
                        for i, frame in enumerate(st.session_state.trajectory_data[1:]):
                            all_xyz += GaussianParser.create_xyz_string(frame)

                        st.sidebar.download_button(
                            label="download trajectory file",
                            data=all_xyz,
                            file_name="complete_trajectory.xyz",
                            mime="chemical/x-xyz",
                            width='stretch'
                        )
                    except Exception as e:
                        st.error(f"Export trajectory failure: {e}")

            # export energy data
            if st.button("📈 Export energies (CSV)", width='stretch'):
                if st.session_state.energy_data:
                    try:
                        df = pd.DataFrame(st.session_state.energy_data)
                        csv_data = df.to_csv(index=False)
                        st.sidebar.download_button(
                            label="download CSV file",
                            data=csv_data,
                            file_name="energy_data.csv",
                            mime="text/csv",
                            width='stretch'
                        )
                    except Exception as e:
                        st.error(f"Export energy data failure: {e}")


        if st.sidebar.button("Clean", type='secondary',width='stretch'):
            st.session_state.parsed = False


def main():
    # setup sidebar
    PageSetUp.render_sidebar()

    # main area
    if st.session_state.parsed:
        # Trajectory player
        with st.expander("### 🎞️ View Optimization", expanded=True):
            if st.session_state.parsed and st.session_state.trajectory_data:
                MoleculeVisualizer.display_trajectory_player_with_3d(
                    trajectory_data=st.session_state.trajectory_data,
                    current_frame=st.session_state.current_frame,
                    total_frames=st.session_state.total_frames,
                    style=st.session_state.view_style,
                    width=st.session_state.view_width,
                    height=st.session_state.view_height,
                    rotation_enabled=st.session_state.rotation_enabled,
                    rotation_axis=st.session_state.get('rotation_axis', 'y'),
                    playback_speed=st.session_state.get('playback_speed', 10),
                )
            else:
                st.info("Please click Decipher first.")

        # Vibration player
        if st.session_state.get('vib_mode_to_play', None) is not None:
            try:
                mode = st.session_state.get('vib_mode_to_play')
                amp = st.session_state.get('vib_amplitude', 0.2)
                period = st.session_state.get('vib_period', 1.5)
                show_arrows = st.session_state.get('vib_show_arrows', True)
                loop = st.session_state.get('vib_loop', True)
                fps = st.session_state.get('vib_fps', 24)
                is_playing = st.session_state.get('vib_playing', True)
                arrow_length = st.session_state.get('vib_arrow_length', 2.0)
                arrow_radius = st.session_state.get('vib_arrow_radius', 0.05)

                base_atoms = st.session_state.trajectory_data[-1]

                with st.expander("### 🎞️ View Vibration", expanded=True):
                    vib_container = st.container()
                    with vib_container:
                        MoleculeVisualizer.display_vibration_mode(
                            atoms=base_atoms,
                            mode=mode,
                            amplitude=amp,
                            period=period,
                            show_arrows=show_arrows,
                            loop=loop,
                            width=st.session_state.get('view_width', 800),
                            height=st.session_state.get('view_height', 500),
                            fps=fps,
                            is_playing=is_playing,
                            arrow_length=arrow_length,
                            arrow_radius=arrow_radius,
                        )

            except Exception as e:
                st.error(f"Oops: {e}")

        if not st.session_state.parsed_gjf:
            with st.expander("##### 📈 Optimization", expanded=True):

                charts = OptimizationMonitor.create_horizontal_charts(
                    st.session_state.energy_data,
                    st.session_state.gradient_data,
                    st.session_state.force_data,
                    st.session_state.rms_force_data,
                    st.session_state.displacement_data,
                    st.session_state.rms_displacement_data,
                    st.session_state.current_frame,
                    st.session_state.irc_rx_coord_data,
                )

                if not st.session_state.parsed_irc:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if charts[0]:
                            st.plotly_chart(charts[0], width='stretch', config={'displayModeBar': False})
                    with col2:
                        if charts[1]:
                            st.plotly_chart(charts[1], width='stretch', config={'displayModeBar': False})
                    with col3:
                        if charts[2]:
                            st.plotly_chart(charts[2], width='stretch', config={'displayModeBar': False})

                    col4, col5, col6 = st.columns(3)
                    with col4:
                        if charts[3]:
                            st.plotly_chart(charts[3], width='stretch', config={'displayModeBar': False})
                    with col5:
                        if charts[4]:
                            st.plotly_chart(charts[4], width='stretch', config={'displayModeBar': False})
                    with col6:
                        if charts[5]:
                            st.plotly_chart(charts[5], width='stretch', config={'displayModeBar': False})

                    st.markdown('</div>', unsafe_allow_html=True)

                elif st.session_state.parsed_irc:
                    if charts[0]:
                        st.plotly_chart(charts[0], width='stretch', config={'displayModeBar': False})

                else:
                    st.info("Generating...")

        # Info cards
        with st.expander("📋 More Details"):

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧬 Info Summary", "📊 Energies", "🧪 Geometry", "📋 Opt",
                                                    "📋 Convergence?", "📋 Thermochemistry"])

            with tab1:
                if st.session_state.trajectory_data and not st.session_state.parsed_gjf:
                        current_atoms = st.session_state.trajectory_data[st.session_state.current_frame]
                        stoichiometry = st.session_state.stoichiometry
                        num_atoms = st.session_state.num_atoms
                        method = st.session_state.method
                        # basis_set = st.session_state.basis_set
                        charge = st.session_state.charges
                        multiplicity = st.session_state.multiplicity
                        GaussianParser.create_molecule_info_table(current_atoms, stoichiometry, num_atoms, method,
                                                                  charge, multiplicity)

                if st.session_state.parsed_gjf:
                    current_atoms = st.session_state.trajectory_data[st.session_state.current_frame]
                    GaussianParser.create_input_info_table(current_atoms)

            with tab2:
                if st.session_state.energy_data:
                    try:
                        df_energy = pd.DataFrame(st.session_state.energy_data)
                        for col in ['energy', 'energy_kcal']:
                            if col in df_energy.columns:
                                df_energy[col] = pd.to_numeric(df_energy[col], errors='coerce')
                        if st.session_state.irc_rx_coord_data:
                            df_energy['RxCoord'] = st.session_state.irc_rx_coord_data
                            df_energy = df_energy.reindex(columns=['step', 'RxCoord','energy', 'energy_kcal'])

                        st.dataframe(df_energy, width='stretch', hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error occurred while displaying energy data: {e}")

            with tab3:
                if st.session_state.trajectory_data and st.session_state.current_frame < len(
                        st.session_state.trajectory_data):
                    try:
                        current_atoms = st.session_state.trajectory_data[st.session_state.current_frame]
                        df_atoms = pd.DataFrame(current_atoms)[['element', 'x', 'y', 'z']]
                        for col in ['x', 'y', 'z']:
                            if col in df_atoms.columns:
                                df_atoms[col] = pd.to_numeric(df_atoms[col], errors='coerce')

                        st.dataframe(df_atoms, width='stretch', hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error occurred while displaying molecule: {e}")


            with tab4:
                if st.session_state.force_data:
                    try:
                        df_opt = pd.DataFrame(st.session_state.force_data)[['step', 'forces', ]]
                        df_opt['rms force'] = pd.DataFrame(st.session_state.rms_force_data)[['rms_force', ]]
                        df_opt['displacement'] = pd.DataFrame(st.session_state.displacement_data)[['displacements']]
                        df_opt['rms displacement'] = pd.DataFrame(st.session_state.rms_displacement_data)[
                            ['rms_displacements', ]]

                        st.dataframe(df_opt, width='stretch', hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error occurred: {e}")


            with tab5:
                if st.session_state.force_data:
                    try:
                        df_converge = pd.DataFrame(st.session_state.force_data)[['step']]
                        df_converge['force'] = pd.DataFrame(st.session_state.force_data)[['converged', ]]
                        df_converge['rms force'] = pd.DataFrame(st.session_state.rms_force_data)[['converged', ]]
                        df_converge['displacement'] = pd.DataFrame(st.session_state.displacement_data)[['converged']]
                        df_converge['rms displacement'] = pd.DataFrame(st.session_state.rms_displacement_data)[
                            ['converged', ]]
                        styled_df_converge = df_converge.style.applymap(
                            lambda x: 'background-color: #A7DAB5; color: black' if x == 'YES' else '',
                        )

                        st.dataframe(styled_df_converge, width='stretch', hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error occurred: {e}")

            with tab6:
                if st.session_state.sum_tcg:
                    temperature = str(st.session_state.temperature)
                    pressure = str(st.session_state.pressure)
                    electronic_energy = str(f"{st.session_state.energy_data[-1]['energy']:.6f}")
                    zpe = str(st.session_state.zpe)
                    tce = str(st.session_state.tce)
                    tch = str(st.session_state.tch)
                    tcg = str(st.session_state.tcg)
                    sum_zpe = str(st.session_state.sum_zpe)
                    sum_tce = str(st.session_state.sum_tce)
                    sum_tch = str(st.session_state.sum_tch)
                    sum_tcg = str(st.session_state.sum_tcg)

                    GaussianParser.creat_thermo_info_table(temperature, pressure, electronic_energy, zpe, tce, tch, tcg,
                                       sum_zpe, sum_tce, sum_tch, sum_tcg)

    else:
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
import streamlit as st
from typing import List, Dict
import json


class MoleculeVisualizer:
    @staticmethod
    def display_trajectory_player_with_3d(trajectory_data: List[List[Dict]], current_frame: int,
                                          total_frames: int, style: str = 'stick',
                                          width: int = None, height: int = None,
                                          rotation_enabled: bool = False, rotation_axis: str = 'y',
                                          playback_speed: int = 10,
                                          ):
        """
        use py3Dmol to show molecule
        """

        if not trajectory_data:
            st.warning("Trajectroy data is unavailable")
            return

        width = width or st.session_state.get('view_width', 800)
        height = height or st.session_state.get('view_height', 500)

        try:
            # construct frame data
            frames_data = []
            for frame_idx, frame_atoms in enumerate(trajectory_data):
                frame_info = {
                    'frame': frame_idx,
                    'atoms': []
                }
                for atom in frame_atoms:
                    frame_info['atoms'].append({
                        'element': atom['element'],
                        'x': float(atom['x']),
                        'y': float(atom['y']),
                        'z': float(atom['z']),
                        'atomic_number': atom.get('atomic_number', 0)
                    })
                frames_data.append(frame_info)

            # convert to JSON
            trajectory_json = json.dumps(frames_data)

            # frames playing state
            is_playing = st.session_state.get('is_playing', False)

            # ---------- initialize ----------
            if "current_frame" not in st.session_state:
                # start from 1 cuz I made a mask frame
                st.session_state.current_frame = 1

            if "is_playing" not in st.session_state:
                st.session_state.is_playing = False

            max_frame = total_frames

            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 6, 1, 1])

            with col2:
                if st.button("⏮️", use_container_width=True):
                    st.session_state.current_frame = 1
                    st.session_state.is_playing = False

                    st.rerun()

            with col3:
                if st.button("◀️", use_container_width=True):
                    st.session_state.current_frame = max(1, st.session_state.current_frame -1)
                    st.session_state.is_playing = False

                    st.rerun()

            with col5:
                if st.button("▶️", use_container_width=True):
                    st.session_state.current_frame = min(max_frame, st.session_state.current_frame + 1)
                    st.session_state.is_playing = False

                    st.rerun()

            with col6:
                if st.button("⏭️", use_container_width=True):
                    st.session_state.current_frame = max_frame
                    st.session_state.is_playing = False

                    st.rerun()

            with col4:
                st.slider(
                    "Frame control",
                    # min_value=0,
                    min_value=1,
                    max_value=max_frame if max_frame > 1 else 2,
                    key="current_frame",
                    label_visibility="collapsed",
                    format="%d",

                )

            # frame control plane
            with col1:
                pass
                st.markdown(f"""
                 <div style="text-align: left; margin: 5px 0;">
                     <span style="font-weight: bold; font-size: 20px;">
                         Step {current_frame} / {total_frames}
                     </span>
                 </div>
                 """, unsafe_allow_html=True)

            # frame control button
            play_col1, play_col2, play_col3, play_col4 = st.columns([2, 2, 6, 2])
            with play_col2:
                if st.button("⏯️ play/stop", use_container_width=True, key="play_pause_js"):
                    st.session_state.is_playing = not st.session_state.is_playing
                    st.rerun()

            with play_col3:
                new_speed = st.slider(
                    "play speed",
                    min_value=10,
                    max_value=30,
                    value=st.session_state.get('playback_speed', 10),
                    key="speed_slider_js",
                    format="%d",
                    label_visibility="collapsed",
                )
                if new_speed != st.session_state.get('playback_speed', 10):
                    st.session_state.playback_speed = new_speed
                    st.rerun()

            with play_col1:
                st.markdown(f"""
                <div style="text-align: left; margin: 5px 0;">
                    <span style="font-weight: normal; font-size: 18px;">
                        Play Trajectory 
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with play_col4:
                st.markdown(f"""
                <div style="text-align: left; margin: 5px 0;">
                    <span style="font-weight: normal; font-size: 18px;">
                        {st.session_state.playback_speed} frames/s 
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # HTML/JavaScript
            style_map = {
                'stick': 'stick',
                'sphere': 'sphere',
                'ballstick': 'ballstick',
                'line': 'line'
            }

            html = f"""
            <div class="mol-viewer-container">
                <div class="unified-3d-wrapper" style="width:{width}px; height:{height}px;">
                    <div id="traj_viewer" style="width:100%; height:100%;"></div>
                </div>
            </div>

            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <script>
            (function(){{
                const trajectoryData = {trajectory_json};
                const currentFrame = {current_frame};
                const totalFrames = {total_frames};
                const styleType = '{style_map.get(style, 'stick')}';
                const rotationEnabled = {str(rotation_enabled).lower()};
                const rotationAxis = '{str(rotation_axis).lower()}';
                const playbackSpeed = {playback_speed};
                let isPlaying = {str(is_playing).lower()};

                // creat 3D viewer
                const viewer = $3Dmol.createViewer(document.getElementById('traj_viewer'), {{
                    backgroundColor: 'white',
                    width: '{width}px',
                    height: '{height}px'
                }});

                // setup style
                let styleConfig = {{}};
                if (styleType === 'stick') {{
                    styleConfig = {{stick: {{radius: 0.15}}}};
                }} else if (styleType === 'sphere') {{
                    styleConfig = {{sphere: {{radius: 0.5}}}};
                }} else if (styleType === 'ballstick') {{
                    styleConfig = {{
                        sphere: {{radius: 0.3}},
                        stick: {{radius: 0.1}}
                    }};
                }} else if (styleType === 'line') {{
                    styleConfig = {{line: {{}}}};
                }}

                // updateDisplay frame
                function updateDisplay(frameIndex) {{
                    if (frameIndex < 0 || frameIndex >= trajectoryData.length) return;

                    const frame = trajectoryData[frameIndex];
                    let xyzString = frame.atoms.length + "\\nGenerated by PuGaussView\\n";

                    frame.atoms.forEach(atom => {{
                        xyzString += atom.element + " " + 
                                   atom.x.toFixed(6) + " " + 
                                   atom.y.toFixed(6) + " " + 
                                   atom.z.toFixed(6) + "\\n";
                    }});

                    viewer.removeAllModels();
                    viewer.addModel(xyzString, "xyz");
                    viewer.setStyle({{}}, styleConfig);

                    // if rotation enabled
                    if (rotationEnabled) {{
                        viewer.spin(axis=rotationAxis, speed=0.5);
                    }} else {{
                        viewer.spin(false);
                    }}

                    viewer.zoomTo();
                    viewer.render();
                }}

                // initial display
                updateDisplay(currentFrame);

                // auto play
                let playInterval = null;
                let currentPlayFrame = currentFrame;

                function startPlayback() {{
                    if (playInterval) clearInterval(playInterval);

                    const intervalTime = 1000 / playbackSpeed; // ms

                    playInterval = setInterval(() => {{
                        currentPlayFrame++;

                        if (currentPlayFrame >= totalFrames) {{
                            currentPlayFrame = 1;
                        }}

                        updateDisplay(currentPlayFrame);

                        // update frame via iframe message

                    }}, intervalTime);
                }}

                function stopPlayback() {{
                    if (playInterval) {{
                        clearInterval(playInterval);
                        playInterval = null;
                    }}
                }}

                // play control
                if (isPlaying) {{
                    startPlayback();
                }} else {{
                    stopPlayback();
                }}


            }})();
            </script>
            """

            # show HTML
            st.components.v1.html(html, height=height + 20, scrolling=False)


        except Exception as e:
            st.error(f"Error occurred: {e}")

    @staticmethod
    def display_vibration_mode(atoms: List[Dict], mode: Dict, amplitude: float = 0.2, period: float = 1.5,
                               show_arrows: bool = True, loop: bool = True, width: int = None, height: int = None,
                               fps: int = 24, is_playing: bool = True, arrow_length: float = 2,
                               arrow_radius: float = 0.05, ):

        if not atoms or not mode or not mode.get('vectors'):
            st.error("Vector unavailable")
            return

        base_atoms = atoms
        n_atoms = len(base_atoms)
        vectors = mode.get('vectors', [])

        if len(vectors) < n_atoms:
            vectors = vectors + [(0.0, 0.0, 0.0)] * (n_atoms - len(vectors))
        elif len(vectors) > n_atoms:
            vectors = vectors[:n_atoms]

        base_coords = [[float(a['x']), float(a['y']), float(a['z'])] for a in base_atoms]
        elements = [a['element'] for a in base_atoms]
        vecs = [[float(v[0]), float(v[1]), float(v[2])] for v in vectors]

        width = width or st.session_state.get('view_width', 800)
        height = height or st.session_state.get('view_height', 500)

        data = {
            'base_coords': base_coords,
            'elements': elements,
            'vecs': vecs,
            'amplitude': float(amplitude),
            'period': float(period),
            'show_arrows': bool(show_arrows),
            'loop': bool(loop),
            'freq': float(mode.get('freq', 0.0)),
            'fps': int(fps),
            'is_playing': bool(is_playing),
            'length': float(arrow_length),
            'radius': float(arrow_radius),
        }
        json_data = json.dumps(data)

        html = f"""
        <div class="mol-viewer-container">
            <div class="unified-3d-wrapper" style="width:{width}px; height:{height}px;">
                <div id="vib_viewer" style="width:100%; height:100%;"></div>
            </div>
        </div>

        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
        (function(){{
            const dat = {json_data};
            const base = dat.base_coords;
            const vecs = dat.vecs;
            const els = dat.elements;
            const amp = dat.amplitude;
            const period = dat.period;
            const show_arrows = dat.show_arrows;
            const loop = dat.loop;
            const fps = dat.fps;
            const length = dat.length;
            const radius = dat.radius;
            let is_playing = dat.is_playing;

            const viewer = $3Dmol.createViewer(document.getElementById('vib_viewer'), {{
                backgroundColor: 'white',
                width: '{width}px',
                height: '{height}px'
            }});

            function coords_to_xyz(coords) {{
                let s = "";
                s += coords.length + "\\nGenerated by Gaussian Visualizer\\n";
                for (let i=0;i<coords.length;i++) {{
                    s += els[i] + " " + coords[i][0].toFixed(6) + " " + coords[i][1].toFixed(6) + " " + coords[i][2].toFixed(6) + "\\n";
                }}
                return s;
            }}

            let coords = JSON.parse(JSON.stringify(base));
            viewer.addModel(coords_to_xyz(coords), "xyz");
            viewer.setStyle({{}}, {{stick: {{radius:0.15}}, sphere: {{radius:0.3}}}});
            viewer.zoomTo();
            viewer.render();

            // get atom object
            const model = viewer.getModel();
            const atoms = (model && model.atoms) ? model.atoms: [];

            function draw_arrows(coords, vectors, length) {{
                viewer.removeAllShapes();
                for (let i=0;i<coords.length;i++) {{
                    const start = {{x: coords[i][0], y: coords[i][1], z: coords[i][2]}};
                    const vec = vectors[i];
                    const end = {{x: coords[i][0] + vec[0]*length, y: coords[i][1] + vec[1]*length, z: coords[i][2] + vec[2]*length}};
                    viewer.addArrow({{start: start, end: end, radius: radius, mid:0.8,color: 'red'}});
                }}
            }}

            let t0 = Date.now();
            let pause_time = 0;
            let anim_id = null;

            function step() {{
                if (!is_playing) return;

                const now = Date.now();
                const elapsed = (now - t0 - pause_time)/1000.0;
                let phase = Math.sin(2*Math.PI*elapsed/period);

                let newcoords = [];
                for (let i=0;i<base.length;i++) {{
                    const x = base[i][0] + vecs[i][0]*amp*phase;
                    const y = base[i][1] + vecs[i][1]*amp*phase;
                    const z = base[i][2] + vecs[i][2]*amp*phase;
                    newcoords.push([x,y,z]);
                }}

                viewer.removeAllModels();
                viewer.addModel(coords_to_xyz(newcoords), "xyz");
                viewer.setStyle({{}}, {{stick: {{radius:0.15}}, sphere: {{radius:0.3}}}});

                if (show_arrows) {{
                    draw_arrows(newcoords, vecs, length);
                }} else {{
                    viewer.removeAllShapes();
                }}

                //viewer.zoomTo();
                viewer.render();
            }}

            anim_id = setInterval(step, 1000 / fps);

        }})();
        </script>
        """

        st.components.v1.html(html, height=height + 20, scrolling=False)


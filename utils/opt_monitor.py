import plotly.graph_objects as go
import streamlit as st
import pandas as pd


class OptimizationMonitor:

    @staticmethod
    def create_single_chart(title, x_data, y_data, x_title, y_title, chart_type='line', current_step=None,
                            color='blue'):

        try:
            fig = go.Figure()

            if chart_type == 'line':
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    name=title
                ))
            elif chart_type == 'scatter':
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(color=color, size=8),
                    name=title
                ))

            # since we got a mask frame, the energy index = frame index - 1
            if current_step is not None and len(x_data) > 0:
                if current_step < len(x_data)+1:
                    current_x = x_data[current_step-1]
                    current_y = y_data[current_step-1]

                    # add vertical line
                    fig.add_vline(
                        x=current_x,
                        line_dash="dash",
                        line_color="red",
                        line_width=1,
                        annotation_text=f"Step {current_step}",
                        annotation_position="top right"
                    )

                    # add a marker to show which step
                    fig.add_trace(go.Scatter(
                        x=[current_x],
                        y=[current_y],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='circle-open'),
                        name='Current Step'
                    ))

            # update layout
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
                title_text=title,
                title_font_size=14,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

            fig.update_xaxes(title_text=x_title, showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text=y_title, showgrid=True, gridwidth=1, gridcolor='LightGray')

            return fig
        except Exception as e:
            st.error(f"创建图表时出错: {e}")
            return None

    @staticmethod
    def create_horizontal_charts(energy_data, gardient_data, force_data, rms_force_data,
                                 displacement_data, rms_displacement_data, current_frame,
                                 rx_coord_data):

        if not energy_data:
            return []

        charts = []

        try:
            df_energy = pd.DataFrame(energy_data)
            if not st.session_state.parsed_irc:

                # 1. Total Energy
                if len(df_energy) > 0:
                    fig1 = OptimizationMonitor.create_single_chart(
                        title="📈 Total Energy",
                        x_data=df_energy['step'],
                        y_data=df_energy['energy'],
                        x_title="Optimization Step",
                        y_title="Total Energy (Hartree)",
                        chart_type='line',
                        current_step=current_frame,
                        color='blue'
                    )
                    if fig1:
                        charts.append(fig1)

                # 2. Maximum Force
                if force_data and len(force_data) > 0:
                    df_force = pd.DataFrame(force_data)
                    fig2 = OptimizationMonitor.create_single_chart(
                        title="⚡ Maximum Interal Force",
                        x_data=df_force['step'],
                        y_data=df_force['forces'],
                        x_title="Optimization Step",
                        y_title="Maximum Interal Force",
                        chart_type='line',
                        current_step=current_frame,
                        color='orange'
                    )
                    if fig2:
                        charts.append(fig2)

                # 3. Maximum Displacement
                if displacement_data and len(displacement_data) > 0:
                    df_disp = pd.DataFrame(displacement_data)
                    fig3 = OptimizationMonitor.create_single_chart(
                        title="🎯 Maximum Interal Displacement",
                        x_data=df_disp['step'],
                        y_data=df_disp['displacements'],
                        x_title="Optimization Step",
                        y_title="Maximum Interal Displacement",
                        chart_type='line',
                        current_step=current_frame,
                        color='orange'
                    )
                    if fig3:
                        charts.append(fig3)

                # 4. RMS Gradient Norm
                if displacement_data and len(gardient_data) > 0:
                    df_grad = pd.DataFrame(gardient_data)
                    fig4 = OptimizationMonitor.create_single_chart(
                        title="📈 RMS Gradient Norm",
                        x_data=df_grad['step'],
                        y_data=df_grad['gradients'],
                        x_title="Optimization Step",
                        y_title="RMS Gradient Norm",
                        chart_type='line',
                        current_step=current_frame,
                        color='purple'
                    )
                    if fig4:
                        charts.append(fig4)

                # 5. RMS Force
                if displacement_data and len(rms_force_data) > 0:
                    df_rms_force = pd.DataFrame(rms_force_data)
                    fig5 = OptimizationMonitor.create_single_chart(
                        title="⚡ RMS Interal Force",
                        x_data=df_rms_force['step'],
                        y_data=df_rms_force['rms_force'],
                        x_title="Optimization Step",
                        y_title="RMS Force",
                        chart_type='line',
                        current_step=current_frame,
                        color='purple'
                    )
                    if fig5:
                        charts.append(fig5)

                # 6. RMS Force
                if displacement_data and len(rms_displacement_data) > 0:
                    df_rms_disp = pd.DataFrame(rms_displacement_data)
                    fig6 = OptimizationMonitor.create_single_chart(
                        title="🎯 RMS Interal Force",
                        x_data=df_rms_disp['step'],
                        y_data=df_rms_disp['rms_displacements'],
                        x_title="Optimization Step",
                        y_title="RMS Force",
                        chart_type='line',
                        current_step=current_frame,
                        color='purple'
                    )
                    if fig6:
                        charts.append(fig6)

                # always 6 charts
                while len(charts) < 6:
                    charts.append(None)
            else:
                if len(rx_coord_data) > 0:
                    df_rx_coord = rx_coord_data
                    fig7 = OptimizationMonitor.create_single_chart(
                        title="📈 Total Energy along IRC",
                        x_data=df_rx_coord,
                        y_data=df_energy['energy'],
                        x_title="Intrinsic Reaction Coordinate",
                        y_title="Total Energy (Hartree)",
                        chart_type='line',
                        current_step=current_frame,
                        color='blue'
                    )
                else:
                    fig7 = OptimizationMonitor.create_single_chart(
                        title="📈 Total Energy",
                        x_data=df_energy['step'],
                        y_data=df_energy['energy'],
                        x_title="Optimization Step",
                        y_title="Total Energy (Hartree)",
                        chart_type='line',
                        current_step=current_frame,
                        color='blue'
                    )
                if fig7:
                    charts.append(fig7)


        except Exception as e:
            st.error(f"Error occurred when generating charts: {e}")

        return charts


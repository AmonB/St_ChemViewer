import streamlit as st
import os

st.set_page_config(page_title="File Browser", page_icon="📁", layout="wide")
st.title("📁 File Browser")


# initialize

if 'working_dir' not in st.session_state:
    st.session_state.working_dir = os.path.expanduser("~")

if 'path' not in st.session_state:
    st.session_state.path = st.session_state.working_dir

if 'last_input' not in st.session_state:
    st.session_state.last_input = st.session_state.path

if "read_file_path" not in st.session_state:
    st.session_state.read_file_path = None


# 路径输入和跳转逻辑
col1, col2, col3 = st.columns([9, 1, 1])
with col1:
    current_input = st.text_input(
        "input_path",
        value=st.session_state.path,
        placeholder="input path and press enter",
        label_visibility="collapsed"
    )
    # check change
    if current_input != st.session_state.last_input:
        st.session_state.last_input = current_input
        # trigger jump-to
        st.session_state.check_jump = True

with col2:
    # jump-to button
    if st.button("↵", width='stretch'):
        st.session_state.check_jump = True

with col3:
    if st.button("🔍", width='stretch', help='Go Analysis page'):
        st.switch_page("pages/gaussian.py")

# check jump-to
if st.session_state.get('check_jump', False):
    if os.path.exists(current_input) and os.path.isdir(current_input):
        if current_input != st.session_state.path:
            st.session_state.path = current_input
            st.session_state.working_dir = st.session_state.path
            st.rerun()
    else:
        st.error("Invalid path")

    # reset check-jump-to status
    st.session_state.check_jump = False



st.markdown(f"**Directory:** `{st.session_state.path}`")

# get directory
current = st.session_state.path
items = []

# add  ..
if os.path.dirname(current) != current:
    items.append(("..", "parent"))

try:
    for item in os.listdir(current):
        full_path = os.path.join(current, item)
        items.append((item, "dir" if os.path.isdir(full_path) else "file"))
except:
    st.error("Can't read directory")

parent_items = [name for name, type_ in items if type_ == "parent"]
dir_items = [name for name, type_ in items if type_ == "dir"]
file_items = [name for name, type_ in items if type_ == "file"]

dir_items = parent_items + dir_items

if dir_items:
    cols = st.columns(3)
    for i, name in enumerate(dir_items):
        with cols[i % 3]:
            if i ==0:
                if st.button(f"⬆️ .. ", key=f"up_{name}", width='stretch'):
                    st.session_state.path = os.path.dirname(current)
                    st.session_state.working_dir = st.session_state.path
                    st.rerun()
            else:
                if st.button(f"📁 {name}", key=f"dir_{name}", width='stretch',):
                    st.session_state.path = os.path.join(current, name)
                    st.session_state.working_dir = st.session_state.path
                    st.rerun()


st.markdown("**Files:**")

if file_items:
    cols = st.columns(3)
    for i, name in enumerate(file_items):

        ext = os.path.splitext(name)[1].lower()
        file_icon = {
            '.txt': '📄', '.log': '📝', '.md': '📖',
            '.py': '🐍', '.js': '📜', '.html': '🌐',
            '.css': '🎨', '.json': '📋', '.xml': '📄',
            '.csv': '📊', '.xlsx': '📊', '.pdf': '📕',
            '.jpg': '🖼️', '.png': '🖼️', '.gif': '🖼️',
            '.mp3': '🎵', '.mp4': '🎬', '.zip': '📦',
            '.exe': '⚙️', '.doc': '📘', '.docx': '📘'
        }.get(ext, '📄')
        with cols[i % 3]:
            if st.button(f"{file_icon} {name}", key=f"file_{name}", width='stretch'):
                st.session_state.read_file_path = os.path.join(current, name)
                st.switch_page("pages/file_reader.py")


with st.sidebar.expander("💡 Tips", expanded=True):
    st.markdown("""
    - Click the directory button to change the path or input your path 
    - Click the file button to show the content
    """)
st.sidebar.markdown('</div>', unsafe_allow_html=True)




# other version
# ------------------------------------------------------------------
# 当前路径显示
# col1, col2 = st.columns([2, 8])
# with col1:
#     st.markdown(f"**Directory:** `{st.session_state.path}`")
# with col2:
#     st.markdown("**Files:**")
#
# # 获取和显示目录内容
# current = st.session_state.path
# items = []
#
# # 添加..
# if os.path.dirname(current) != current:
#     items.append(("..", "parent"))
#
# try:
#     for item in os.listdir(current):
#         full_path = os.path.join(current, item)
#         items.append((item, "dir" if os.path.isdir(full_path) else "file"))
# except:
#     st.error("无法读取目录")
#
# # 显示所有项目
# col1, col2 = st.columns([2, 8])
# with col1:
#     if st.button(f"⬆️ .. ", key=f"up_..", width='stretch'):
#         st.session_state.path = os.path.dirname(current)
#         st.session_state.working_dir = st.session_state.path
#         st.rerun()
#
#
#     for name, type_ in items:
#         if type_ == "dir":
#             if st.button(f"📁 {name}", key=f"dir_{name}", width='stretch',):
#                 st.session_state.path = os.path.join(current, name)
#                 st.session_state.working_dir = st.session_state.path
#                 st.rerun()

# with col2:
#     for name, type_ in items:
#         if type_ == "file":
#             st.markdown(f"📄 {name}")


# with col2:
#     cols = st.columns(2)
#     for i, item in enumerate(items):
#         with cols[i % 2]:
#             if list(item)[1] == "file":
#                 name = list(item)[0]
#                 if st.button(f"📄 {name}", key=f"file_{name}", width='stretch'):
#                     st.session_state.read_file_path = os.path.join(current, name)
#                     st.switch_page("pages/file_reader.py")




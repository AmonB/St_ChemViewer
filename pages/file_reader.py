import streamlit as st
import os

st.set_page_config(page_title="File Reader", page_icon="📄", layout="wide")
st.title("📄 File Reader")

def file_viewer(file_path):

    # st.markdown(f"## 📄 File Reader")

    try:

        ext = os.path.splitext(file_path)[1].lower()

        # txt file types
        text_extensions = ['.txt', '.log', '.md', '.py', '.js', '.html',
                           '.css', '.json', '.xml', '.csv', '.yaml', '.yml',
                           '.gjf', '.com', '.out', '.ini']

        if ext in text_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 代码高亮
                code_languages = {
                    '.py': 'python',
                    '.js': 'javascript',
                    '.html': 'html',
                    '.css': 'css',
                    '.json': 'json',
                    '.xml': 'xml',
                    '.yaml': 'yaml',
                    '.yml': 'yaml',

                }

                language = code_languages.get(ext, 'text')

                # show line numbers
                # lines = content.split('\n')
                # st.markdown(f"**Lines:** {len(lines)}")

                st.code(content, language=language)

            except UnicodeDecodeError:
                # if not UTF-8, try another
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                    st.code(content, language='text')
                except:
                    st.info("Can't preview in text format")
                    # preview on hexadecimal system
                    with open(file_path, 'rb') as f:
                        bytes_content = f.read(1024)  # read first 1KB
                        st.text("Preview first 1KB on hexadecimal system:")
                        st.code(bytes_content.hex())
            except Exception as e:
                st.error(f"Can't read file: {str(e)}")

        # picture
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            try:
                from PIL import Image
                image = Image.open(file_path)
                st.image(image, caption=os.path.basename(file_path))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Size: {image.size}")
                with col2:
                    st.write(f"Format: {image.format}")
                with col3:
                    st.write(f"Mode: {image.mode}")
            except Exception as e:
                st.error(f"Can't load picture: {str(e)}")

        # other types
        else:
            st.info("Unsupported file type")

            # try binary system
            try:
                with open(file_path, 'rb') as f:
                    # preview first 1KB
                    preview_bytes = f.read(1024)
                    st.text("Preview first 1KB:")
                    st.code(
                        preview_bytes.hex()[:500] + "..." if len(preview_bytes.hex()) > 500 else preview_bytes.hex())
            except Exception as e:
                st.error(f"Can't read file: {str(e)}")

    except Exception as e:
        st.error(f"Open file error: {str(e)}")




# initial
if "read_file_path" not in st.session_state:
    st.session_state.read_file_path = None


# sidebar
with st.sidebar.expander("📄 File Reader", expanded=True):
    if st.session_state.read_file_path is None:
        st.markdown(f"##### Current file is: `{st.session_state.read_file_path}`, please choose a file from File Browser!")


    else:
        file_path = st.session_state.read_file_path
        if not os.path.exists(file_path):
            st.error("File does not exist!")
        else:
            file_size = os.path.getsize(file_path)
            st.markdown(f"**File:** `{file_path}`")
            st.caption(f"Size: {file_size / 1024:.1f} kb")
            st.caption(f"Type: {os.path.splitext(file_path)[1]}" or "without extension")


    if st.button("Back to 📁 File Browser", type='secondary', width='stretch'):
        st.switch_page('pages/file_browser.py')



# main display area
if st.session_state.read_file_path is not None:
    file_viewer(file_path)


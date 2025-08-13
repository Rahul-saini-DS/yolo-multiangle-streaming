import streamlit as st

st.session_state.setdefault('step', 5)
st.title('Task Configuration')

st.markdown('''
AI model and task assignment (demo)

> **Task configuration interface coming soon.** In production, you'll configure:
> - Primary/Secondary model selection
> - Task types (detect/segment/classify/pose/obb)
> - Confidence thresholds
> - Output formatting
''')

if st.button('Start Monitoring'):
    st.session_state['step'] = 6

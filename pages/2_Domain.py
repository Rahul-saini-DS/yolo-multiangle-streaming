import streamlit as st

st.session_state.setdefault('step', 2)
st.title('Select Industry Domain')

domains = [
    {'key': 'retail', 'label': 'Retail'},
    {'key': 'health', 'label': 'Health'},
    {'key': 'manufacturing', 'label': 'Manufacturing'},
    {'key': 'transport', 'label': 'Transport'},
    {'key': 'sports', 'label': 'Sports'},
]

selected = st.session_state.get('selected_domain', 'retail')

tabs = st.tabs([d['label'] for d in domains])
for i, domain in enumerate(domains):
    with tabs[i]:
        if st.button(f"Select {domain['label']}", key=f"domain_{domain['key']}"):
            st.session_state['selected_domain'] = domain['key']
            st.session_state['step'] = 3

st.markdown('---')
if st.session_state.get('selected_domain'):
    st.success(f"Selected: {st.session_state['selected_domain'].title()}")
    if st.button('Next'):
        st.session_state['step'] = 3

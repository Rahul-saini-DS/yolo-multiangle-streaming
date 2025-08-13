import streamlit as st

st.session_state.setdefault('step', 1)
st.title('Choose Deployment Area')

areas = [
    {'key': 'in-store', 'label': 'In-Store', 'desc': 'Customer-facing retail areas'},
    {'key': 'back-office', 'label': 'Back-Office', 'desc': 'Staff and administrative areas'},
    {'key': 'warehouse', 'label': 'Warehouse', 'desc': 'Storage and logistics'},
    {'key': 'parking-lot', 'label': 'Parking Lot', 'desc': 'Outdoor customer areas'},
]

selected = st.session_state.get('selected_area', 'in-store')

cols = st.columns(len(areas))
for i, area in enumerate(areas):
    with cols[i]:
        if st.button(area['label'], key=f"area_{area['key']}"):
            st.session_state['selected_area'] = area['key']
            st.session_state['step'] = 2
        st.caption(area['desc'])

st.markdown('---')
if st.session_state.get('selected_area'):
    st.success(f"Selected: {st.session_state['selected_area'].replace('-', ' ').title()}")
    if st.button('Next'):
        st.session_state['step'] = 2

import streamlit as st

st.session_state.setdefault('step', 3)
st.title('Solution Catalog')

solutions = [
    {'key': 'ai-vision', 'label': 'AI Vision Inspection', 'desc': 'Automated quality control and defect detection'},
    {'key': 'customer-behavior', 'label': 'Customer Behavior Analysis', 'desc': 'Track movement patterns, dwell time, and engagement'},
    {'key': 'visitor-profile', 'label': 'Visitor Profile Analysis', 'desc': 'Demographic insights and customer segmentation'},
    {'key': 'product-engagement', 'label': 'Product Engagement', 'desc': 'Monitor product interactions and shelf performance'},
    {'key': 'customer-service', 'label': 'Customer Service Quality', 'desc': 'Evaluate service interactions and response times'},
    {'key': 'customer-experience', 'label': 'Customer Experience', 'desc': 'Sentiment analysis and satisfaction tracking'},
    {'key': 'inventory', 'label': 'Inventory Management', 'desc': 'Real-time stock levels and shelf monitoring'},
    {'key': 'outside-analytics', 'label': 'Outside & In-Store Analytics', 'desc': 'Comprehensive foot traffic and space utilization'},
    {'key': 'checkout', 'label': 'Checkout Experience', 'desc': 'Queue management and transaction optimization'},
    {'key': 'security', 'label': 'Security', 'desc': 'Intrusion detection and suspicious activity alerts'},
]

selected = st.session_state.get('selected_solutions', set())

cols = st.columns(3)
for i, sol in enumerate(solutions):
    with cols[i % 3]:
        checked = sol['key'] in selected
        if st.checkbox(sol['label'], value=checked, key=f"sol_{sol['key']}"):
            selected.add(sol['key'])
        else:
            selected.discard(sol['key'])
        st.caption(sol['desc'])
st.session_state['selected_solutions'] = selected

st.markdown('---')
if selected:
    st.success(f"Selected: {', '.join([s.title().replace('-', ' ') for s in selected])}")
    if st.button('Next'):
        st.session_state['step'] = 4

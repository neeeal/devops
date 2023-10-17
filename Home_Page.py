import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

# Add a CSS style to center the title and make it bigger
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column; /* Stack the emoji and title */
        height: 80vh;
    }

    .bigger-title {
        font-size: 56px; /* You can adjust the font size as needed */
    }
    .emoji {
        font-size: 72px; /* Adjust the size of the emoji as needed */
    }
    .description {
        text-align: center;
        font-size: 18px; /* Adjust the font size as needed */
        margin-top: 20px; /* Add some margin space for better spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a container div with the center class to center the emoji, title, and description
st.markdown(
    '<div class="center"><span class="emoji">🕓</span><h1 class="bigger-title">Time Classification</h1>'
    '<p class="description">Simplify Analog Clock Reading.</p></div>',
    unsafe_allow_html=True,
)

st.sidebar.success("Select page above.")

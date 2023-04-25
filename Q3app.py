import streamlit as st
from streamlit import components

def main():
    st.title("Delta Selectbox and Score")

    score = 85
    options = ["-5%", "-2%", "2%", "5%"]
    selected_option = st.selectbox("Delta:", options)

    delta_value = float(selected_option.strip("%"))
    color = "green" if delta_value >= 0 else "red"
    arrow = "▲" if delta_value >= 0 else "▼"

    st.write(f"Score: {score}")
    st.write(
        components.v1.html(
            f'<p style="color:{color};">{arrow} {selected_option}</p>',
            width=200,
            height=50,
        ),
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

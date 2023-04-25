import streamlit as st
from streamlit import components

def main():
    st.title("Delta Selectbox and Score")

    score = 85
    options = ["-5%", "-2%", "2%", "5%"]
    selected_option = st.selectbox("Delta:", options)

    delta_value = float(selected_option.strip("%"))
    color = "green" if delta_value >= 0 else "red"
    st.write(
        components.v1.html(
            f'<p style="color:{color}; display:inline-block; margin-left:10px;">{selected_option}</p>',
            width=200,
            height=50,
        ),
        unsafe_allow_html=True,
    )

    updated_score = score * (1 + delta_value / 100)
    st.write(f"Original Score: {score}")
    st.write(f"Updated Score: {updated_score:.2f}")

if __name__ == "__main__":
    main()

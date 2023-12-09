import streamlit as st

st.set_page_config(page_title="About", page_icon="📈")
st.sidebar.header("About")
st.title("About the Developer")

def bio_page():

    

    st.image("./vipul_pic.jpg", caption="Vipul Jain", use_column_width=True) # 'Project_Final/image.jpg'

    st.markdown(
        """
        Hello there! 👋 I'm Vipul Jain, the developer behind StockPredictor. 
        As a passionate data scientist and AI Developer, I strive to create 
        innovative solutions that make complex data accessible and actionable.

        **Background:**

        I pursuing a degree in Masters in Data Science from Michigan State University. My journey into 
        data science began with a curiosity to uncover patterns and insights in data, 
        and it has since evolved into a commitment to building tools that empower 
        individuals in their decision-making processes.

        **Skills:**
        - Data Science
        - Machine Learning and Deep Learning
        - Optimization: Mixed Integer Programming, Combinatorial Optimization
        - Quantum Computing
        - Python, SQL, etc.


        **Publications:**
        - **Patent** *(Pending)*: **A METHOD AND SYSTEM OF ENHANCED HYBRID QUANTUM-CLASSICAL COMPUTING MECHANISM FOR SOLVING OPTIMIZATION PROBLEMS**\n
        - **Flyer** : Quantum Based Carbon Capture & Sequestration Solution [here](https://www.infosys.com/services/incubating-emerging-technologies/documents/sequestration-solution.pdf)

        **Contact:**
        Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/vipul-jain-737125159/) or 
        [GitHub](https://github.com/vipuljain-vinyl). I'm always open to collaboration and 
        exploring new opportunities.

        **Resume:**
        You can find my detailed resume [here](https://drive.google.com/file/d/1Qfb2EiFtreuLGHyj--S5CR84y-ckwg8c/view?usp=drive_link).

        Happy exploring with StockPredictor! 🚀

        """
    )

bio_page()

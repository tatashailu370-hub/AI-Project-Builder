import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Project Builder",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT DATABASE
# =========================================================

PROJECTS = [

    {
        "project": "AI Based Student Performance Prediction",
        "domain": "Education",
        "type": "Prediction",
        "algorithm": "Random Forest / Logistic Regression",
        "dataset": "Student performance dataset",
        "description": "Predict student academic performance using attendance, marks, study hours and previous academic records."
    },

    {
        "project": "AI Based Road Pothole Detection",
        "domain": "Transportation",
        "type": "Computer Vision",
        "algorithm": "CNN / Transfer Learning",
        "dataset": "Road pothole image dataset",
        "description": "Detect potholes from road images."
    },

    {
        "project": "Smart Traffic Prediction System",
        "domain": "Transportation",
        "type": "Prediction",
        "algorithm": "Random Forest / Regression",
        "dataset": "Traffic flow dataset",
        "description": "Predict traffic congestion using historical traffic data."
    },

    {
        "project": "Fake News Detection System",
        "domain": "Cyber Security",
        "type": "NLP",
        "algorithm": "TF-IDF + Logistic Regression",
        "dataset": "Fake news dataset",
        "description": "Classify online news articles as real or potentially fake."
    },

    {
        "project": "Spam Email Detection",
        "domain": "Cyber Security",
        "type": "Classification",
        "algorithm": "Naive Bayes",
        "dataset": "Spam email dataset",
        "description": "Detect whether an email or message is spam."
    },

    {
        "project": "Phishing Website Detection",
        "domain": "Cyber Security",
        "type": "Classification",
        "algorithm": "Random Forest",
        "dataset": "Phishing website dataset",
        "description": "Identify suspicious websites using URL features."
    },

    {
        "project": "Heart Disease Risk Prediction",
        "domain": "Healthcare",
        "type": "Classification",
        "algorithm": "Random Forest / Logistic Regression",
        "dataset": "Heart disease dataset",
        "description": "Estimate heart disease risk using health-related attributes."
    },

    {
        "project": "Diabetes Prediction System",
        "domain": "Healthcare",
        "type": "Classification",
        "algorithm": "Logistic Regression / Random Forest",
        "dataset": "Diabetes dataset",
        "description": "Predict diabetes risk using health information."
    },

    {
        "project": "Personalized Learning Recommendation System",
        "domain": "Education",
        "type": "Recommendation",
        "algorithm": "Content-Based Filtering",
        "dataset": "Course and student preference dataset",
        "description": "Recommend learning resources according to student interests and performance."
    },

    {
        "project": "College Placement Prediction",
        "domain": "Education",
        "type": "Prediction",
        "algorithm": "Random Forest",
        "dataset": "College placement dataset",
        "description": "Predict placement probability using academic performance and skills."
    },

    {
        "project": "House Price Prediction",
        "domain": "Finance",
        "type": "Regression",
        "algorithm": "Linear Regression / Random Forest",
        "dataset": "House price dataset",
        "description": "Predict house prices using property features."
    },

    {
        "project": "Loan Approval Prediction",
        "domain": "Finance",
        "type": "Classification",
        "algorithm": "Random Forest",
        "dataset": "Loan approval dataset",
        "description": "Predict whether a loan application is likely to be approved."
    },

    {
        "project": "Customer Churn Prediction",
        "domain": "Business",
        "type": "Classification",
        "algorithm": "Logistic Regression / Random Forest",
        "dataset": "Customer churn dataset",
        "description": "Predict whether a customer may leave a service."
    },

    {
        "project": "Sales Forecasting System",
        "domain": "Business",
        "type": "Prediction",
        "algorithm": "Random Forest / Regression",
        "dataset": "Historical sales dataset",
        "description": "Forecast future sales using historical sales data."
    },

    {
        "project": "Customer Segmentation",
        "domain": "Business",
        "type": "Clustering",
        "algorithm": "K-Means",
        "dataset": "Customer transaction dataset",
        "description": "Group customers according to purchasing behavior."
    },

    {
        "project": "Crop Disease Detection",
        "domain": "Agriculture",
        "type": "Computer Vision",
        "algorithm": "CNN / Transfer Learning",
        "dataset": "Plant leaf image dataset",
        "description": "Identify crop diseases from plant leaf images."
    },

    {
        "project": "Crop Yield Prediction",
        "domain": "Agriculture",
        "type": "Prediction",
        "algorithm": "Random Forest / Regression",
        "dataset": "Crop yield dataset",
        "description": "Predict crop yield using soil and weather information."
    },

    {
        "project": "Smart Irrigation Prediction",
        "domain": "Agriculture",
        "type": "Prediction",
        "algorithm": "Random Forest",
        "dataset": "Soil moisture and weather dataset",
        "description": "Predict irrigation requirements."
    },

    {
        "project": "Air Quality Prediction",
        "domain": "Environment",
        "type": "Prediction",
        "algorithm": "Random Forest / Regression",
        "dataset": "Air quality dataset",
        "description": "Predict air pollution levels."
    },

    {
        "project": "Rainfall Prediction System",
        "domain": "Environment",
        "type": "Prediction",
        "algorithm": "Random Forest",
        "dataset": "Weather dataset",
        "description": "Predict rainfall using historical weather conditions."
    },

    {
        "project": "Smart Parking Detection",
        "domain": "IoT",
        "type": "Computer Vision",
        "algorithm": "CNN / Image Classification",
        "dataset": "Parking lot image dataset",
        "description": "Detect available and occupied parking spaces."
    },

    {
        "project": "Smart Home Energy Monitoring",
        "domain": "IoT",
        "type": "Data Analysis",
        "algorithm": "Pandas + Matplotlib",
        "dataset": "Energy consumption dataset",
        "description": "Analyze household energy consumption."
    },

    {
        "project": "Movie Recommendation System",
        "domain": "Entertainment",
        "type": "Recommendation",
        "algorithm": "Content-Based Filtering",
        "dataset": "MovieLens dataset",
        "description": "Recommend movies based on user preferences."
    },

    {
        "project": "Product Review Sentiment Analysis",
        "domain": "E-Commerce",
        "type": "NLP",
        "algorithm": "TF-IDF + Logistic Regression",
        "dataset": "Product review dataset",
        "description": "Classify product reviews as positive, negative or neutral."
    },

    {
        "project": "Food Recommendation System",
        "domain": "Food",
        "type": "Recommendation",
        "algorithm": "Content-Based Filtering",
        "dataset": "Food and nutrition dataset",
        "description": "Recommend food based on preferences and nutrition."
    },

    {
        "project": "Sports Player Performance Prediction",
        "domain": "Sports",
        "type": "Prediction",
        "algorithm": "Random Forest",
        "dataset": "Sports statistics dataset",
        "description": "Predict player performance using historical statistics."
    }

]


df = pd.DataFrame(PROJECTS)


# =========================================================
# DOMAIN KEYWORDS
# =========================================================

DOMAIN_KEYWORDS = {

    "Education": [
        "student", "school", "college", "exam",
        "marks", "attendance", "learning",
        "teacher", "education", "placement",
        "course", "academic", "study"
    ],

    "Healthcare": [
        "health", "disease", "hospital",
        "patient", "doctor", "medical",
        "diabetes", "heart", "medicine",
        "healthcare", "clinic"
    ],

    "Cyber Security": [
        "security", "spam", "phishing",
        "malware", "fraud", "attack",
        "intrusion", "password", "cyber",
        "fake news", "hacking"
    ],

    "Finance": [
        "loan", "bank", "stock", "finance",
        "insurance", "credit", "investment",
        "money", "price", "financial"
    ],

    "Agriculture": [
        "crop", "farm", "agriculture",
        "plant", "soil", "irrigation",
        "yield", "farmer", "farming",
        "leaf"
    ],

    "IoT": [
        "iot", "sensor", "smart home",
        "device", "automation",
        "temperature", "monitoring",
        "energy", "parking"
    ],

    "Transportation": [
        "traffic", "vehicle", "road",
        "accident", "transport",
        "parking", "route",
        "pothole", "driving",
        "congestion"
    ],

    "Business": [
        "business", "customer", "sales",
        "employee", "company", "startup",
        "marketing", "churn"
    ],

    "E-Commerce": [
        "shopping", "product",
        "ecommerce", "review",
        "online store"
    ],

    "Environment": [
        "air", "water", "pollution",
        "weather", "rainfall",
        "flood", "waste",
        "environment", "solar",
        "electricity"
    ],

    "Food": [
        "food", "restaurant",
        "recipe", "cooking",
        "meal", "nutrition"
    ],

    "Sports": [
        "sports", "player", "match",
        "football", "cricket",
        "basketball", "performance"
    ],

    "Entertainment": [
        "movie", "music", "song",
        "book", "entertainment"
    ]

}


# =========================================================
# PROJECT TYPE KEYWORDS
# =========================================================

TYPE_KEYWORDS = {

    "Classification": [
        "classification",
        "classify",
        "detect",
        "detection",
        "identify",
        "spam",
        "fraud",
        "disease",
        "fake",
        "recognition"
    ],

    "Regression": [
        "price",
        "salary",
        "cost",
        "amount",
        "regression",
        "value"
    ],

    "Prediction": [
        "predict",
        "prediction",
        "forecast",
        "estimate",
        "risk"
    ],

    "Recommendation": [
        "recommend",
        "recommendation",
        "suggest",
        "personalized",
        "similar"
    ],

    "Clustering": [
        "cluster",
        "group",
        "segmentation",
        "segment",
        "grouping"
    ],

    "NLP": [
        "text",
        "language",
        "sentiment",
        "chatbot",
        "email",
        "news",
        "review",
        "voice",
        "document",
        "nlp"
    ],

    "Computer Vision": [
        "image",
        "face",
        "video",
        "object",
        "vision",
        "gesture",
        "camera",
        "photo",
        "pothole",
        "leaf"
    ],

    "Data Analysis": [
        "analysis",
        "analytics",
        "dashboard",
        "visualization",
        "report",
        "insights"
    ]

}


# =========================================================
# DETECT DOMAIN
# =========================================================

def detect_domain(text):

    text = text.lower()

    scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():

        scores[domain] = sum(
            1 for word in keywords
            if word in text
        )

    best_domain = max(
        scores,
        key=scores.get
    )

    if scores[best_domain] == 0:
        return "Data Science"

    return best_domain


# =========================================================
# DETECT PROJECT TYPE
# =========================================================

def detect_type(text):

    text = text.lower()

    scores = {}

    for project_type, keywords in TYPE_KEYWORDS.items():

        scores[project_type] = sum(
            1 for word in keywords
            if word in text
        )

    best_type = max(
        scores,
        key=scores.get
    )

    if scores[best_type] == 0:
        return "Data Analysis"

    return best_type


# =========================================================
# RECOMMEND PROJECTS
# =========================================================

def recommend_projects(
    project_idea,
    top_n=5
):

    if not project_idea.strip():

        return pd.DataFrame()

    documents = (

        df["project"]
        + " "
        + df["domain"]
        + " "
        + df["type"]
        + " "
        + df["algorithm"]
        + " "
        + df["dataset"]
        + " "
        + df["description"]

    )

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b"
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        user_vector = vectorizer.transform(
            [project_idea]
        )

        scores = cosine_similarity(
            user_vector,
            matrix
        )[0]

    except Exception:

        scores = np.zeros(
            len(df)
        )

    results = df.copy()

    results["score"] = scores

    detected_domain = detect_domain(
        project_idea
    )

    detected_type = detect_type(
        project_idea
    )

    results.loc[
        results["domain"] == detected_domain,
        "score"
    ] += 0.20

    results.loc[
        results["type"] == detected_type,
        "score"
    ] += 0.15

    results = (
        results
        .sort_values(
            "score",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return results


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🤖 AI Project Builder"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔨 Project Builder",
        "🔎 Smart Recommendation",
        "📚 Project Database",
        "ℹ️ About"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🤖 AI Project Builder'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Turn your project idea into a complete AI / ML project plan'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "💡 **Enter Any Idea**\n\n"
            "Describe your project in simple words."
        )

    with col2:

        st.success(
            "🧠 **Smart Analysis**\n\n"
            "Automatically detects domain and project type."
        )

    with col3:

        st.warning(
            "🚀 **Build Blueprint**\n\n"
            "Get algorithm, dataset and workflow."
        )

    st.divider()

    st.subheader(
        "Example Project Ideas"
    )

    examples = [

        "AI based road pothole detection",

        "Student performance prediction",

        "Fake news detection",

        "Crop disease detection",

        "Movie recommendation system",

        "Heart disease prediction",

        "Smart traffic prediction",

        "Customer churn prediction"

    ]

    for example in examples:

        st.write(
            "• " + example
        )


# =========================================================
# PROJECT BUILDER
# =========================================================

elif page == "🔨 Project Builder":

    st.markdown(
        '<div class="main-title">'
        '🔨 Build Your Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter any AI, ML or Data Science project idea'
        '</div>',
        unsafe_allow_html=True
    )

    project_idea = st.text_area(

        "💡 Enter your project idea",

        placeholder=(
            "Example: AI based system for "
            "detecting road potholes"
        ),

        height=130

    )

    col1, col2, col3 = st.columns(3)

    with col1:

        domain = st.selectbox(

            "📊 Select Domain",

            [
                "Auto Detect",
                "Education",
                "Healthcare",
                "Cyber Security",
                "Finance",
                "Agriculture",
                "IoT",
                "Transportation",
                "Business",
                "E-Commerce",
                "Environment",
                "Food",
                "Sports",
                "Entertainment"
            ]

        )

    with col2:

        level = st.selectbox(

            "📚 Project Level",

            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]

        )

    with col3:

        project_type = st.selectbox(

            "⚙️ Project Type",

            [
                "Auto Detect",
                "Classification",
                "Regression",
                "Prediction",
                "Recommendation",
                "Clustering",
                "NLP",
                "Computer Vision",
                "Data Analysis"
            ]

        )

    st.write("")

    build = st.button(

        "🚀 BUILD MY PROJECT",

        use_container_width=True

    )

    if build:

        if not project_idea.strip():

            st.error(
                "Please enter a project idea."
            )

        else:

            final_domain = (

                detect_domain(project_idea)

                if domain == "Auto Detect"

                else domain

            )

            final_type = (

                detect_type(project_idea)

                if project_type == "Auto Detect"

                else project_type

            )

            st.session_state[
                "project_result"
            ] = {

                "idea": project_idea,

                "domain": final_domain,

                "type": final_type,

                "level": level

            }

            st.success(
                "✅ Project successfully generated!"
            )

    if "project_result" in st.session_state:

        result = st.session_state[
            "project_result"
        ]

        st.divider()

        st.header(
            "🎯 Project Blueprint"
        )

        st.subheader(
            "📌 " + result["idea"]
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Domain",
                result["domain"]
            )

        with c2:

            st.metric(
                "Project Type",
                result["type"]
            )

        with c3:

            st.metric(
                "Level",
                result["level"]
            )

        st.subheader(
            "🎯 Project Objective"
        )

        st.write(

            "Develop an AI/Data Science solution for "
            + result["idea"]
            + ". The system processes relevant data, "
              "applies a suitable machine learning approach "
              "and provides useful results through a web interface."

        )

        algorithm_map = {

            "Classification":
                "Logistic Regression / Random Forest",

            "Prediction":
                "Random Forest / Regression",

            "Regression":
                "Linear Regression / Random Forest",

            "Recommendation":
                "Content-Based Recommendation",

            "Clustering":
                "K-Means Clustering",

            "NLP":
                "TF-IDF + Logistic Regression",

            "Computer Vision":
                "CNN / Transfer Learning",

            "Data Analysis":
                "Pandas + Matplotlib"

        }

        algorithm = algorithm_map.get(

            result["type"],

            "Scikit-learn Machine Learning"

        )

        st.subheader(
            "🧠 Recommended Algorithm"
        )

        st.info(
            algorithm
        )

        st.subheader(
            "📂 Suggested Dataset"
        )

        dataset_map = {

            "Education":
                "Student / education dataset",

            "Healthcare":
                "Healthcare dataset",

            "Transportation":
                "Traffic / transportation dataset",

            "Agriculture":
                "Agriculture / crop dataset",

            "Finance":
                "Financial dataset",

            "Cyber Security":
                "Cyber security dataset",

            "IoT":
                "IoT sensor dataset",

            "Business":
                "Business / customer dataset",

            "Environment":
                "Environmental dataset"

        }

        dataset = dataset_map.get(

            result["domain"],

            "Suitable public dataset from Kaggle or UCI"

        )

        st.info(
            dataset
        )

        st.subheader(
            "✨ Main Features"
        )

        features = [

            "User-friendly Streamlit interface",

            "Input validation",

            "Data preprocessing",

            "Exploratory Data Analysis",

            "Machine Learning model",

            "Prediction / recommendation output",

            "Performance evaluation",

            "Graphs and visualizations"

        ]

        for feature in features:

            st.write(
                "✅ " + feature
            )

        st.subheader(
            "🔄 Project Workflow"
        )

        workflow = [

            "Collect dataset",

            "Clean missing or incorrect values",

            "Perform exploratory data analysis",

            "Prepare features",

            "Train machine learning model",

            "Evaluate model",

            "Build Streamlit interface",

            "Display results"

        ]

        for i, step in enumerate(
            workflow,
            start=1
        ):

            st.write(
                f"**{i}.** {step}"
            )

        st.subheader(
            "🛠️ Technology Stack"
        )

        st.write(
            "Python • Streamlit • Pandas • "
            "NumPy • Scikit-learn • Matplotlib"
        )


# =========================================================
# SMART RECOMMENDATION
# =========================================================

elif page == "🔎 Smart Recommendation":

    st.markdown(
        '<div class="main-title">'
        '🔎 Smart Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter any project idea and find similar projects."
    )

    idea = st.text_area(

        "Enter project idea",

        placeholder=(
            "Example: AI system to predict student marks"
        ),

        height=120

    )

    if st.button(

        "🔍 FIND SIMILAR PROJECTS",

        use_container_width=True

    ):

        if not idea.strip():

            st.warning(
                "Please enter a project idea."
            )

        else:

            detected_domain = detect_domain(
                idea
            )

            detected_type = detect_type(
                idea
            )

            st.success(
                f"Detected Domain: **{detected_domain}**"
            )

            st.info(
                f"Detected Type: **{detected_type}**"
            )

            results = recommend_projects(
                idea,
                5
            )

            for i, row in results.iterrows():

                score = int(
                    min(
                        100,
                        max(
                            0,
                            row["score"] * 100
                        )
                    )
                )

                st.markdown(
                    f"### {i + 1}. {row['project']}"
                )

                st.write(
                    "**Domain:**",
                    row["domain"]
                )

                st.write(
                    "**Type:**",
                    row["type"]
                )

                st.write(
                    "**Algorithm:**",
                    row["algorithm"]
                )

                st.write(
                    "**Dataset:**",
                    row["dataset"]
                )

                st.write(
                    "**Match Score:**",
                    f"{score}%"
                )

                st.caption(
                    row["description"]
                )

                st.divider()


# =========================================================
# PROJECT DATABASE
# =========================================================

elif page == "📚 Project Database":

    st.markdown(
        '<div class="main-title">'
        '📚 Project Database'
        '</div>',
        unsafe_allow_html=True
    )

    selected_domain = st.selectbox(

        "Filter by Domain",

        ["All"]
        + sorted(
            df["domain"].unique()
        )

    )

    if selected_domain == "All":

        display_df = df

    else:

        display_df = df[
            df["domain"]
            == selected_domain
        ]

    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True

    )

    st.write(
        f"Total Projects: **{len(display_df)}**"
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">'
        'ℹ️ About AI Project Builder'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        AI Project Builder is a free student-focused application
        that helps users convert simple project ideas into
        structured AI, Machine Learning and Data Science projects.
        """
    )

    st.subheader(
        "🔧 Technologies Used"
    )

    for technology in [

        "Python",

        "Streamlit",

        "Pandas",

        "NumPy",

        "Scikit-learn",

        "Matplotlib"

    ]:

        st.write(
            "✅ " + technology
        )

    st.subheader(
        "💰 Cost"
    )

    st.success(
        "No paid AI API is required."
    )

    st.subheader(
        "🚀 Main Capabilities"
    )

    capabilities = [

        "Automatic domain detection",

        "Automatic project type detection",

        "Smart project recommendation",

        "Algorithm recommendation",

        "Dataset suggestion",

        "Project workflow",

        "Technology stack",

        "Project database"

    ]

    for item in capabilities:

        st.write(
            "• " + item
        )